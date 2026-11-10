"""iPhone DeviceMotion CSV -> AI-IMU pickle format.

CSV conventions (W3C DeviceMotion / browser-style logging):
  - timestamp: ms
  - rotationRate_{alpha,beta,gamma}: deg/s; empirically body rates are
    [wx,wy,wz] = deg2rad([gamma, beta, alpha]) (matches attitude derivative)
  - accelerationIncludingGravity_{x,y,z}: m/s^2 (WITH gravity); negated on ingest
    to match KITTI/OXTS specific-force sign for IEKF (R(a-b)+g)
  - acceleration_{x,y,z}: user acceleration WITHOUT gravity (unused here)
  - gps_{lat,lon,speed,course}: WGS84 deg, m/s, deg from true north (-1 invalid)
  - xTrueNorthZVertical_quat_*: game_rv / attitude, device -> NWU (x true north, z up);
    converted to ENU. Used for phone->car roll/pitch; GPS velocity supplies mount yaw.

IEKF / KITTI expect u = [wx, wy, wz, ax, ay, az] in rad/s and m/s^2 (specific force),
nav frame with z-up (g = [0,0,-9.81]), v_gt = [ve, vn, vu], p_gt local ENU.
"""
import os
import csv
from collections import OrderedDict

import numpy as np
import torch
from termcolor import cprint
from navpy import lla2ned

from dataset import BaseDataset
from utils_numpy_filter import NUMPYIEKF


# xTrueNorthZVertical is (x north, y west, z up); IEKF / p_gt / v_gt use ENU.
_ROT_NWU_TO_ENU = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
MAX_MOUNT_TILT_DEG = 15.0
MAX_HANDLING_GYRO = 0.5  # rad/s
HANDLING_WINDOW_S = 30.0


def _quat_to_rotmat(qx, qy, qz, qw):
    """Body->nav rotation from unit quaternion (x, y, z, w)."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _read_csv_columns(csv_path):
    """Load only the columns we need (memory-friendly for large files)."""
    wanted = [
        'timestamp',
        'accelerationIncludingGravity_x', 'accelerationIncludingGravity_y',
        'accelerationIncludingGravity_z',
        'rotationRate_alpha', 'rotationRate_beta', 'rotationRate_gamma',
        'gps_lat', 'gps_lon', 'gps_speed', 'gps_course',
        'xTrueNorthZVertical_quat_x', 'xTrueNorthZVertical_quat_y',
        'xTrueNorthZVertical_quat_z', 'xTrueNorthZVertical_quat_w',
        'barometer_altitude',
    ]
    cols = {k: [] for k in wanted}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        missing = [k for k in wanted if k not in reader.fieldnames]
        if missing:
            raise KeyError('CSV missing columns: {}'.format(missing))
        for row in reader:
            for k in wanted:
                v = row[k]
                cols[k].append(float(v) if v not in ('', None) else np.nan)
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _estimate_mount(ang_gt, v_gt, min_speed=3.0):
    """Phone->car rotation (car: x forward, y left, z up).

    game_rv / xTrueNorthZVertical (in ang_gt) already maps phone->ENU, so it undoes
    roll and pitch relative to world-vertical. Car is assumed approximately level in ENU;
    GPS velocity supplies car yaw so the remaining phone-vs-car heading is observed.
    """
    moving = np.linalg.norm(v_gt[:, :2], axis=1) > min_speed
    if not np.any(moving):
        raise RuntimeError('No moving samples for mount estimation')
    ang = ang_gt[moving]
    vel = v_gt[moving]
    fwd_enu = vel.copy()
    fwd_enu[:, 2] = 0.0
    fwd_enu /= np.linalg.norm(fwd_enu, axis=1, keepdims=True)
    up_enu = np.array([0.0, 0.0, 1.0])
    left_enu = np.cross(np.broadcast_to(up_enu, fwd_enu.shape), fwd_enu)
    # Columns of Rot_n_c are car axes expressed in ENU.
    Rot_n_c = np.stack([fwd_enu, left_enu, np.broadcast_to(up_enu, fwd_enu.shape)], axis=2)
    Rot_n_p = np.stack([NUMPYIEKF.from_rpy(*a) for a in ang])
    # phone -> car = (ENU <- car)^T (ENU <- phone)
    R_mean = np.einsum('nji,njk->ik', Rot_n_c, Rot_n_p) / len(ang)
    U, _, Vt = np.linalg.svd(R_mean)
    Rot_c_p = U.dot(Vt)
    if np.linalg.det(Rot_c_p) < 0:
        U[:, -1] *= -1
        Rot_c_p = U.dot(Vt)
    return Rot_c_p


def csv_to_mondict(csv_path, name, estimate_mount=False, mount_rpy_deg=None):
    """Convert one DeviceMotion CSV into the pickle dict used by BaseDataset."""
    cprint('Reading {}'.format(csv_path), 'cyan')
    d = _read_csv_columns(csv_path)

    t_ms = d['timestamp']
    t = (t_ms - t_ms[0]) / 1000.0
    t0 = t_ms[0] / 1000.0

    # Gyro deg/s -> rad/s. Axis map chosen to maximize correlation with
    # dR/dt from xTrueNorthZVertical quaternions (not naive W3C beta/gamma/x/y).
    wx = np.deg2rad(d['rotationRate_gamma'])
    wy = np.deg2rad(d['rotationRate_beta'])
    wz = np.deg2rad(d['rotationRate_alpha'])

    # Specific force WITH gravity. iOS/W3C accelerationIncludingGravity is opposite
    # the OXTS/KITTI convention used by IEKF (R(a-b)+g with g=[0,0,-9.81]):
    # KITTI rest az≈+9.8, iOS face-up az≈-9.8 → negate to match.
    ax = -d['accelerationIncludingGravity_x']
    ay = -d['accelerationIncludingGravity_y']
    az = -d['accelerationIncludingGravity_z']
    u = np.stack([wx, wy, wz, ax, ay, az], axis=1)

    # GPS -> local ENU (z-up), matching KITTI v_gt = [ve, vn, vu]
    lat = d['gps_lat']
    lon = d['gps_lon']
    alt = np.nan_to_num(d['barometer_altitude'], nan=0.0)

    valid_gps = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6)
    if not np.any(valid_gps):
        raise RuntimeError('No valid GPS in {}'.format(csv_path))
    i0 = int(np.argmax(valid_gps))
    lat0, lon0, alt0 = lat[i0], lon[i0], alt[i0]

    # Forward-fill GPS gaps for pose GT
    lat_f, lon_f, alt_f = lat.copy(), lon.copy(), alt.copy()
    last_lat, last_lon, last_alt = lat[i0], lon[i0], alt[i0]
    for i in range(len(lat_f)):
        if np.isfinite(lat_f[i]) and abs(lat_f[i]) > 1e-6 and np.isfinite(lon_f[i]):
            last_lat, last_lon = lat_f[i], lon_f[i]
            if np.isfinite(alt_f[i]):
                last_alt = alt_f[i]
        lat_f[i], lon_f[i], alt_f[i] = last_lat, last_lon, last_alt

    ned = lla2ned(lat_f, lon_f, alt_f, lat0, lon0, alt0,
                  latlon_unit='deg', alt_unit='m', model='wgs84')
    # lla2ned -> N,E,D ; we want E,N,U (z-up)
    p_gt = np.stack([ned[:, 1], ned[:, 0], -ned[:, 2]], axis=1)

    speed = d['gps_speed']
    course = d['gps_course']
    course_rad = np.deg2rad(course)
    valid_vel = (course >= 0) & np.isfinite(speed) & np.isfinite(course)
    ve = np.where(valid_vel, speed * np.sin(course_rad), 0.0)
    vn = np.where(valid_vel, speed * np.cos(course_rad), 0.0)
    vu = np.zeros_like(ve)
    # forward-fill velocity through brief invalids after first fix
    last_v = np.zeros(3)
    v_gt = np.zeros((len(speed), 3))
    for i in range(len(speed)):
        if valid_vel[i]:
            last_v = np.array([ve[i], vn[i], vu[i]])
        v_gt[i] = last_v

    # Attitude GT from true-north reference quaternion
    qx = d['xTrueNorthZVertical_quat_x']
    qy = d['xTrueNorthZVertical_quat_y']
    qz = d['xTrueNorthZVertical_quat_z']
    qw = d['xTrueNorthZVertical_quat_w']
    ang_gt = np.zeros((len(t), 3))
    for i in range(len(t)):
        if not np.isfinite([qx[i], qy[i], qz[i], qw[i]]).all():
            ang_gt[i] = ang_gt[i - 1] if i else 0.0
            continue
        Rot = _ROT_NWU_TO_ENU.dot(_quat_to_rotmat(qx[i], qy[i], qz[i], qw[i]))
        roll, pitch, yaw = NUMPYIEKF.to_rpy(Rot)
        ang_gt[i] = [roll, pitch, yaw]
    # Leading samples without attitude take the first valid one (filter inits from ang_gt[0])
    q_ok = np.isfinite(np.stack([qx, qy, qz, qw], axis=1)).all(axis=1)
    ang_gt[:int(np.argmax(q_ok))] = ang_gt[int(np.argmax(q_ok))]

    # Phones were mounted in different poses; KITTI nets/filter assume a car-aligned IMU,
    # so express u and attitude in the estimated car frame (Rot_c_i only refines small residuals).
    # game_rv supplies roll/pitch vs world-vertical; GPS velocity supplies car yaw.
    if mount_rpy_deg is not None:
        r, p, y = np.deg2rad(mount_rpy_deg)
        Rot_c_p = NUMPYIEKF.from_rpy(r, p, y)
        u = np.concatenate([u[:, :3].dot(Rot_c_p.T), u[:, 3:].dot(Rot_c_p.T)], axis=1)
        ang_gt = np.array([NUMPYIEKF.to_rpy(NUMPYIEKF.from_rpy(*a).dot(Rot_c_p.T)) for a in ang_gt])
        cprint('  phone->car fixed mount rpy (deg): {}'.format(
            np.array(mount_rpy_deg).round(1)), 'green')
    elif estimate_mount:
        Rot_c_p = _estimate_mount(ang_gt, v_gt)
        u = np.concatenate([u[:, :3].dot(Rot_c_p.T), u[:, 3:].dot(Rot_c_p.T)], axis=1)
        ang_gt = np.array([NUMPYIEKF.to_rpy(NUMPYIEKF.from_rpy(*a).dot(Rot_c_p.T)) for a in ang_gt])
        cprint('  phone->car mount rpy (deg): {}'.format(
            np.rad2deg(NUMPYIEKF.to_rpy(Rot_c_p)).round(1)), 'green')
    else:
        cprint('  phone->car mount estimation off', 'yellow')

    # Phone being placed in / removed from the mount at the ends breaks the fixed mount
    # (and the filter inits attitude from ang_gt[0]); drop those segments.
    handling = (np.rad2deg(np.abs(ang_gt[:, :2]).max(axis=1)) > MAX_MOUNT_TILT_DEG) | \
               (np.linalg.norm(u[:, :3], axis=1) > MAX_HANDLING_GYRO)
    head = np.where(handling & (t < t[0] + HANDLING_WINDOW_S))[0]
    tail = np.where(handling & (t > t[-1] - HANDLING_WINDOW_S))[0]
    i_end = int(tail[0]) if len(tail) else len(t)
    i0 = max(i0, int(head[-1]) + 1 if len(head) else 0)
    lla0 = [float(lat_f[i0]), float(lon_f[i0]), float(alt_f[i0])]
    if i_end - i0 < 2:
        raise RuntimeError(
            'Trimming leaves fewer than two usable samples in {}'.format(csv_path))

    # Drop samples before first GPS (origin undefined / parked noise OK but GT bad)
    if i0 > 0 or i_end < len(t):
        cprint('  trimming {} leading / {} trailing samples'.format(i0, len(t) - i_end), 'yellow')
        t = t[i0:i_end] - t[i0]
        u = u[i0:i_end]
        p_gt = p_gt[i0:i_end] - p_gt[i0]
        v_gt = v_gt[i0:i_end]
        ang_gt = ang_gt[i0:i_end]

    dt = np.diff(t)
    cprint(
        '  N={}, duration={:.1f}s, median dt={:.4f}s (~{:.1f} Hz), '
        '|acc|={:.2f}, |gyro|={:.4f} rad/s'.format(
            len(t), t[-1], np.median(dt), 1.0 / np.median(dt),
            np.linalg.norm(u[:, 3:], axis=1).mean(),
            np.linalg.norm(u[:, :3], axis=1).mean()),
        'green')

    return {
        't': torch.from_numpy(t).float(),
        'p_gt': torch.from_numpy(p_gt).float(),
        'ang_gt': torch.from_numpy(ang_gt).float(),
        'v_gt': torch.from_numpy(v_gt).float(),
        'u': torch.from_numpy(u).float(),
        'name': name,
        't0': float(t0),
        'lla0': lla0,
        'estimate_mount': bool(estimate_mount) and mount_rpy_deg is None,
        'mount_rpy_deg': list(mount_rpy_deg) if mount_rpy_deg is not None else None,
    }


class PhoneDataset(BaseDataset):
    """Dataset wrapper; pickles live in path_data_save (no KITTI mix-in)."""

    odometry_benchmark = OrderedDict()

    def __init__(self, args):
        csv_path = getattr(args, 'csv_path', None)
        self.only_sequence = (
            os.path.splitext(os.path.basename(csv_path))[0] if csv_path else None)
        super(PhoneDataset, self).__init__(args)
        # Evaluate every converted sequence end-to-end
        self.odometry_benchmark = OrderedDict(
            (name, [0, None]) for name in self.datasets
        )

    def get_datasets(self):
        self.datasets = []
        if not os.path.isdir(self.path_data_save):
            return
        for f in sorted(os.listdir(self.path_data_save)):
            if not f.endswith(self.pickle_extension):
                continue
            name = f[:-len(self.pickle_extension)]
            if self.only_sequence is not None and name != self.only_sequence:
                continue
            self.datasets.append(name)
        self.divide_datasets()

    @staticmethod
    def read_data(args):
        out_dir = args.path_data_save
        os.makedirs(out_dir, exist_ok=True)
        csv_path = getattr(args, 'csv_path', None)
        if csv_path:
            csv_paths = [csv_path]
        else:
            csv_dir = args.path_data_csv
            csv_paths = [
                os.path.join(csv_dir, f)
                for f in sorted(os.listdir(csv_dir))
                if f.endswith('.csv')
            ]
        if not csv_paths:
            raise FileNotFoundError('No CSV files in {}'.format(args.path_data_csv))
        for csv_path in csv_paths:
            seq_name = os.path.splitext(os.path.basename(csv_path))[0]
            mondict = csv_to_mondict(
                csv_path, seq_name,
                estimate_mount=getattr(args, 'estimate_mount', False),
                mount_rpy_deg=getattr(args, 'mount_rpy_deg', None))
            PhoneDataset.dump(mondict, out_dir, seq_name)
            cprint('  wrote {}{}.p'.format(out_dir + os.sep, seq_name), 'green')
        cprint('Done converting {} sequences'.format(len(csv_paths)), 'green')
