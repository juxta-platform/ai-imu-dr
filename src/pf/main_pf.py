"""Two-stage particle-filter localization.

Stage 1: run the IEKF once on the raw IMU to get velocity.
Stage 2: lightweight PF integrates that velocity with heading_bias and vel_scale,
         constrained by an occupancy map.

    python pf/main_pf.py recording.csv --map map.npy --res 0.05 --init-x 10 --init-y 20
    python pf/main_pf.py recording.csv --map outdoor_walk_occupancy.npy   # UTM map, starts at GPS fix

Run from ``src``. Writes ``results/iphone_zupt_raw/pf/<sequence>_pf.p``, ``.png``, and
``_pf_particles.html``.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import base64
import io
import json
import math
from html import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from main_iphone import PhoneArgs, _pickles_match_mount
from iphone_dataset import PhoneDataset
try:
    from .particle_filter import (
        PfParams, _particle_snapshot, _write_estimate, estimate_cloud, init_particles,
        is_y_down_grid, load_map_metadata, load_occupancy_map, maybe_resample,
        normalized_weights, pf_step)
    from .centreline_pf import (
        RoadDistanceMap, RoadGraph, estimate_road, init_road_particles, pf_step_road)
except ImportError:
    from particle_filter import (
        PfParams, _particle_snapshot, _write_estimate, estimate_cloud, init_particles,
        is_y_down_grid, load_map_metadata, load_occupancy_map, maybe_resample,
        normalized_weights, pf_step)
    from centreline_pf import (
        RoadDistanceMap, RoadGraph, estimate_road, init_road_particles, pf_step_road)
from utils import prepare_data
from utils_numpy_filter import NUMPYIEKF
from utils_torch_filter import prepare_filter


def parse_cli():
    parser = argparse.ArgumentParser(
        description='Two-stage PF: IEKF velocity + lightweight map-matching particles.')
    parser.add_argument('csv', help='iPhone DeviceMotion CSV.')
    parser.add_argument(
        '--map', required=True,
        help='Occupancy .npy. 0 = free, 1 = occupied. Axis 0 is x, axis 1 is y. '
             'A sibling *_metadata.json with grid_cell_size_m sets the resolution '
             'and flips the grid vertically, matching the RoNIN floorplan loader.')
    parser.add_argument(
        '--centreline', default=None,
        help='Centreline graph JSON from make_centreline.py. Particles then move '
             'on the road graph. The occupancy npy is still the background.')
    parser.add_argument('--res', type=float, default=None, help='Cell size in metres.')
    parser.add_argument('--origin-x', type=float, default=0.0)
    parser.add_argument('--origin-y', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.5, help='Values above this are occupied.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--flipud', dest='flipud', action='store_true', help='Force a vertical flip.')
    group.add_argument('--no-flipud', dest='flipud', action='store_false', help='Do not flip.')
    parser.set_defaults(flipud=None)
    parser.add_argument(
        '--init-x', type=float, default=None,
        help='Start x on the map, metres. Defaults to the first GPS fix on a UTM map.')
    parser.add_argument(
        '--init-y', type=float, default=None,
        help='Start y on the map, metres. Defaults to the first GPS fix on a UTM map.')
    parser.add_argument('--init-z', type=float, default=0.0)
    parser.add_argument(
        '--start-s', type=float, default=0.0,
        help='Start this many seconds into the recording, using its attitude, velocity, '
             'and GPS position at that time.')
    parser.add_argument(
        '--yaw-offset-deg', type=float, default=None,
        help='Rotate the recording ENU frame into the map frame, degrees about z. '
             'Defaults to UTM grid convergence on a UTM map, else 0.')
    parser.add_argument('--num-particles', type=int, default=1000)
    parser.add_argument('--init-pos-sigma', type=float, default=1.5, help='Metres.')
    parser.add_argument('--init-yaw-sigma-deg', type=float, default=math.degrees(0.20))
    parser.add_argument('--blocked-penalty', type=float, default=3.0)
    parser.add_argument(
        '--no-slide', action='store_true',
        help='Move particles freely; penalise time spent in occupied cells instead of sliding.')
    parser.add_argument('--wall-penalty', type=float, default=10.0, help='Log-weight per second in occupied cells (--no-slide).')
    parser.add_argument(
        '--road-sigma', type=float, default=8.0,
        help='Metres. Distance-to-road sigma for --centreline with --no-slide.')
    parser.add_argument(
        '--noise-per-s', action='store_true',
        help='Treat walk / position noise sigmas as per sqrt(second) and scale by sqrt(dt).')
    parser.add_argument('--seed', type=lambda text: int(text, 0), default=0x5EED1234)
    parser.add_argument(
        '--estimate-mount', dest='estimate_mount', action='store_true',
        help='Same mount estimate as main_iphone.py.')
    parser.add_argument(
        '--mount-rpy', nargs=3, type=float, metavar=('ROLL', 'PITCH', 'YAW'),
        help='Fixed phone-to-car roll pitch yaw, degrees.')
    parser.add_argument('--no-zupt', action='store_true')
    parser.set_defaults(estimate_mount=False)
    return parser.parse_args()


def load_phone_dataset(args):
    os.makedirs(args.path_data_save, exist_ok=True)
    os.makedirs(args.path_results, exist_ok=True)
    if args.read_data and args.test_sequences:
        existing = {
            os.path.splitext(name)[0]
            for name in os.listdir(args.path_data_save)
            if name.endswith('.p')
        }
        if set(args.test_sequences) <= existing and _pickles_match_mount(args) and \
                _pickles_have_lla0(args):
            print('Pickles already present in {}; reusing them.'.format(args.path_data_save))
            args.read_data = 0
        elif set(args.test_sequences) <= existing:
            print('Reconverting {}: mount setting differs or the start fix (lla0) is missing.'.format(
                args.path_data_save))
    if args.read_data:
        PhoneDataset.read_data(args)
    dataset = PhoneDataset(args)
    if not dataset.datasets:
        raise SystemExit('No sequence pickle for {}'.format(args.test_sequences))
    return dataset


def _pickles_have_lla0(args):
    for name in args.test_sequences:
        if 'lla0' not in PhoneDataset.load(os.path.join(args.path_data_save, name + '.p')):
            return False
    return True


def latlon_to_utm(lat_deg, lon_deg, zone):
    """WGS84 to UTM northern hemisphere (Snyder), metres."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    k0 = 0.9996
    lat = math.radians(lat_deg)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    t = math.tan(lat) ** 2
    c = ep2 * math.cos(lat) ** 2
    aa = math.cos(lat) * (math.radians(lon_deg) - lon0)
    e4 = e2 * e2
    e6 = e4 * e2
    m = a * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat
             - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * lat)
             + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * lat)
             - (35 * e6 / 3072) * math.sin(6 * lat))
    easting = k0 * n * (aa + (1 - t + c) * aa ** 3 / 6
                        + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * aa ** 5 / 120) + 500000.0
    northing = k0 * (m + n * math.tan(lat) * (
        aa * aa / 2 + (5 - t + 9 * c + 4 * c * c) * aa ** 4 / 24
        + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * aa ** 6 / 720))
    return easting, northing


def latlon_to_map_xy(lat_deg, lon_deg, meta):
    """Map frame of a ``y_direction: down`` UTM grid, as loaded: x east, y north."""
    epsg = int(meta['utm_epsg'])
    if not 32601 <= epsg <= 32660:
        raise ValueError('only northern UTM (EPSG 326xx) is supported, got {}'.format(epsg))
    easting, northing = latlon_to_utm(lat_deg, lon_deg, epsg - 32600)
    return (easting - float(meta['utm_origin_easting']),
            northing - float(meta['utm_origin_northing']))


def utm_start_pose(lla0, meta):
    """Start xy on the map and the true-north to grid-north yaw offset."""
    x0, y0 = latlon_to_map_xy(lla0[0], lla0[1], meta)
    x1, y1 = latlon_to_map_xy(lla0[0] + 1e-4, lla0[1], meta)
    yaw = math.atan2(y1 - y0, x1 - x0) - math.pi / 2.0
    return (x0, y0), yaw


def map_frame_pose(ang_gt, v_gt, p_gt, yaw_offset_rad, init_xy, init_z):
    """Rotate the recording's ENU pose into the map frame and place the start on the map."""
    ang0 = np.array(ang_gt[0], dtype=np.float64, copy=True)
    ang0[2] += yaw_offset_rad
    v0 = NUMPYIEKF.rotz(yaw_offset_rad).dot(np.asarray(v_gt[0], dtype=np.float64))
    c = math.cos(yaw_offset_rad)
    s = math.sin(yaw_offset_rad)
    p_map = np.array(p_gt, dtype=np.float64, copy=True)
    p_map[:, 0] = c * p_gt[:, 0] - s * p_gt[:, 1] + init_xy[0]
    p_map[:, 1] = s * p_gt[:, 0] + c * p_gt[:, 1] + init_xy[1]
    p_map[:, 2] = p_gt[:, 2] + init_z
    return ang0, v0, p_map


def rotate_velocity_to_map(v_iekf, yaw_offset_rad):
    """Rotate IEKF world-frame velocity into the map frame."""
    c = math.cos(yaw_offset_rad)
    s = math.sin(yaw_offset_rad)
    v_map = np.empty_like(v_iekf)
    v_map[:, 0] = c * v_iekf[:, 0] - s * v_iekf[:, 1]
    v_map[:, 1] = s * v_iekf[:, 0] + c * v_iekf[:, 1]
    if v_iekf.shape[1] > 2:
        v_map[:, 2] = v_iekf[:, 2]
    return v_map


def plot_pf(path, occ, p_pf, p_kf, p_gt, particles_xy, particles_log_w, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        occ.occupied.astype(float).T, origin='lower', cmap='gray_r',
        vmin=0.0, vmax=1.0, interpolation='nearest',
        extent=[
            occ.origin_x, occ.origin_x + occ.nx * occ.res,
            occ.origin_y, occ.origin_y + occ.ny * occ.res,
        ])
    if p_gt is not None:
        ax.plot(p_gt[:, 0], p_gt[:, 1], color='0.45', lw=1.0, label='gps')
    if p_kf is not None:
        ax.plot(p_kf[:, 0], p_kf[:, 1], color='C1', lw=1.0, label='iekf')
    ax.plot(p_pf[:, 0], p_pf[:, 1], color='C0', lw=1.5, label='pf')
    weights = normalized_weights(particles_log_w)
    ax.scatter(
        particles_xy[:, 0], particles_xy[:, 1], c=weights, s=12, cmap='plasma',
        zorder=4, label='particles')
    xs = [p_pf[:, 0], particles_xy[:, 0]]
    ys = [p_pf[:, 1], particles_xy[:, 1]]
    if p_kf is not None:
        xs.append(p_kf[:, 0])
        ys.append(p_kf[:, 1])
    if p_gt is not None:
        finite = np.isfinite(p_gt[:, 0]) & np.isfinite(p_gt[:, 1])
        xs.append(p_gt[finite, 0])
        ys.append(p_gt[finite, 1])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    pad = 15.0
    ax.set_xlim(float(np.min(x)) - pad, float(np.max(x)) + pad)
    ax.set_ylim(float(np.min(y)) - pad, float(np.max(y)) + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


_ANIMATION_STATES = [
    ['X (m)', 0, 1],
    ['Y (m)', 1, 1],
    ['Heading bias (deg)', 2, 1],      # already in degrees in snapshot
    ['Vel scale', 3, 1],
    ['Blocked-motion EMA', 4, 1],
    ['Log weight', 5, 1],
    ['Normalized weight', 6, 1],
]


def save_particle_html(path, occ, times, p_kf, p_gt, p_pf, frames, title):
    """Offline replay of the particle cloud, same page as the RoNIN filter."""
    buffer = io.BytesIO()
    image = np.flipud(occ.occupied.T.astype(np.float64))
    plt.imsave(buffer, image, format='png', cmap='gray_r', vmin=0, vmax=1, origin='upper')
    map_uri = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    idx = [frame[0] for frame in frames]
    data = {
        'time': (np.asarray(times)[idx] - times[0]).tolist(),
        'input': None if p_kf is None else np.asarray(p_kf)[idx, :2].tolist(),
        'gt': None if p_gt is None else np.asarray(p_gt)[idx, :2].tolist(),
        'estimate': np.asarray(p_pf)[idx, :2].tolist(),
        'frames': [[k, frame[1]] for k, frame in enumerate(frames)],
        'origin': [occ.origin_x, occ.origin_y],
        'size': [occ.nx * occ.res, occ.ny * occ.res],
        'state_specs': _ANIMATION_STATES,
    }
    document = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><title>__TITLE__</title>
<style>body{font:16px sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}
canvas{width:100%;border:1px solid #aaa}input{width:50%}button,select{padding:6px}
table{border-collapse:collapse;width:100%;margin:16px 0}th,td{text-align:right;padding:6px;border-bottom:1px solid #ddd}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px}.charts canvas{min-width:0}
</style><h1>__TITLE__</h1>
<p>Particles: colored by selected state ·
<span style="color:#d62728">Red: PF estimate</span> ·
<span style="color:#1976d2">Blue: IEKF</span> ·
<span style="color:#248c46">Green: GPS</span></p>
<p>x is east and y is north, in metres. Particle dots are the cloud after the map
update and before resampling, not persistent identities.</p>
<p><label>Particle state <select id="state"></select></label>
<span id="scale"></span></p>
<canvas id="map"></canvas><p>
<button id="play">Play</button> <input id="frame" aria-label="Frame" type="range" min="0" value="0">
<select id="speed" aria-label="Playback speed"><option value="1">1×</option>
<option value="5" selected>5×</option><option value="10">10×</option></select>
<span id="status"></span></p>
<div class="charts"><canvas id="histogram" width="540" height="240"></canvas>
<canvas id="history" width="540" height="240"></canvas></div>
<p>Histogram and table describe the displayed sample. History shows its mean and
min–max envelope. Color and plot scales stay fixed across frames.
Weights are normalized over all particles; resampling resets them to uniform.</p>
<table><thead><tr><th>State</th><th>Min</th><th>Mean</th><th>Max</th></tr></thead>
<tbody id="stats"></tbody></table>
<script id="data" type="application/json">__DATA__</script>
<script>
const d=JSON.parse(document.getElementById('data').textContent);
const canvas=document.getElementById('map'), ctx=canvas.getContext('2d');
const slider=document.getElementById('frame'), play=document.getElementById('play');
const speed=document.getElementById('speed'), status=document.getElementById('status');
const stateSelect=document.getElementById('state');
const states=d.state_specs;
states.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s[0];stateSelect.appendChild(o);});
stateSelect.value=2;
const summaries=states.map(s=>d.frames.map(f=>{
  const v=f[1].map(p=>p[s[1]]*s[2]);
  return [Math.min(...v),v.reduce((a,b)=>a+b,0)/v.length,Math.max(...v)];
}));
const limits=summaries.map(series=>{
  let lo=Infinity,hi=-Infinity;for(const s of series){lo=Math.min(lo,s[0]);hi=Math.max(hi,s[2]);}
  if(lo===hi){const pad=Math.max(Math.abs(lo)*0.05,0.001);lo-=pad;hi+=pad;}return [lo,hi];
});
const map=new Image(); slider.max=d.frames.length-1;
let current=0, playing=false, elapsed=0, previous=null;
function point(p){
  return [(p[0]-d.origin[0])/d.size[0]*canvas.width,
          (1-(p[1]-d.origin[1])/d.size[1])*canvas.height];
}
function dot(p,r,color){const q=point(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],r,0,2*Math.PI);ctx.fill();}
function trail(path,k,color){
  if(!path)return;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();
  for(let i=0;i<=k;i++){const q=point(path[i]);if(i===0)ctx.moveTo(...q);else ctx.lineTo(...q);}
  ctx.stroke();dot(path[k],4,color);
}
function color(value,lo,hi){return `hsl(${240*(1-Math.max(0,Math.min(1,(value-lo)/(hi-lo))))},75%,45%)`;}
function diagnostics(selected){
  const s=states[selected], [lo,hi]=limits[selected], series=summaries[selected];
  document.getElementById('scale').textContent=`Blue ${lo.toPrecision(4)} → Red ${hi.toPrecision(4)}`;
  document.getElementById('stats').innerHTML=states.map((spec,i)=>
    `<tr><td>${spec[0]}</td>${summaries[i][current].map(v=>`<td>${v.toPrecision(5)}</td>`).join('')}</tr>`).join('');
  const h=document.getElementById('histogram').getContext('2d');
  h.clearRect(0,0,540,240);h.fillStyle='#222';h.font='13px sans-serif';
  h.fillText(s[0]+' — particle count',12,20);
  const bins=Array(25).fill(0);
  for(const p of d.frames[current][1])bins[Math.min(24,Math.max(0,Math.floor((p[s[1]]*s[2]-lo)/(hi-lo)*25)))]++;
  const peak=Math.max(1,...bins);
  bins.forEach((n,i)=>{h.fillStyle=color(lo+(i+0.5)/25*(hi-lo),lo,hi);h.fillRect(45+i*19,205-n/peak*160,18,n/peak*160);});
  h.fillStyle='#222';h.fillText(String(peak),5,50);h.fillText('0',15,205);
  h.fillText(lo.toPrecision(4),45,230);h.fillText(hi.toPrecision(4),460,230);
  const c=document.getElementById('history').getContext('2d');
  c.clearRect(0,0,540,240);c.font='13px sans-serif';c.fillStyle='#222';
  c.fillText(s[0]+' — mean and min–max',12,20);
  const duration=d.time[d.time.length-1]||1;
  const x=i=>60+450*d.time[d.frames[i][0]]/duration, y=v=>205-160*(v-lo)/(hi-lo);
  c.beginPath();series.forEach((v,i)=>{if(i===0)c.moveTo(x(i),y(v[0]));else c.lineTo(x(i),y(v[0]));});
  for(let i=series.length-1;i>=0;i--)c.lineTo(x(i),y(series[i][2]));
  c.closePath();c.fillStyle='#ddd';c.fill();c.beginPath();
  series.forEach((v,i)=>{if(i===0)c.moveTo(x(i),y(v[1]));else c.lineTo(x(i),y(v[1]));});
  c.strokeStyle='#8050bf';c.lineWidth=2;c.stroke();c.beginPath();
  c.moveTo(x(current),40);c.lineTo(x(current),205);c.strokeStyle='#d62728';c.stroke();
  c.fillStyle='#222';c.fillText(hi.toPrecision(3),2,50);c.fillText(lo.toPrecision(3),2,205);
  c.fillText('0 s',60,230);c.fillText(duration.toFixed(1)+' s',460,230);
}
function draw(){
  const f=d.frames[current], k=f[0];ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.globalAlpha=0.35;ctx.drawImage(map,0,0,canvas.width,canvas.height);ctx.globalAlpha=1;
  trail(d.gt,k,'#248c46');trail(d.input,k,'#1976d2');
  const selected=Number(stateSelect.value), s=states[selected], [lo,hi]=limits[selected];
  ctx.globalAlpha=0.65;for(const p of f[1])dot(p,2,color(p[s[1]]*s[2],lo,hi));ctx.globalAlpha=1;
  trail(d.estimate,k,'#d62728');slider.value=current;
  status.textContent=`${d.time[k].toFixed(1)} s · frame ${current+1}/${d.frames.length}`;
  diagnostics(selected);
}
function tick(now){
  if(playing && previous!==null){
    elapsed+=(now-previous)/1000*Number(speed.value);
    while(current+1<d.frames.length && d.time[d.frames[current+1][0]]<=elapsed)current++;
    draw();if(current===d.frames.length-1){playing=false;play.textContent='Play';}
  }
  previous=now;requestAnimationFrame(tick);
}
play.onclick=()=>{if(current===d.frames.length-1){current=0;elapsed=0;draw();}
  playing=!playing;previous=null;play.textContent=playing?'Pause':'Play';};
slider.oninput=()=>{current=Number(slider.value);elapsed=d.time[d.frames[current][0]];draw();};
stateSelect.onchange=()=>draw();
map.onload=()=>{canvas.width=1100;canvas.height=Math.round(1100*map.height/map.width);
  draw();requestAnimationFrame(tick);};
map.src='__MAP__';
</script></html>'''
    text = (
        document.replace('__TITLE__', escape(title))
        .replace('__DATA__', json.dumps(data, separators=(',', ':'), allow_nan=False))
        .replace('__MAP__', map_uri))
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)


def main():
    cli = parse_cli()
    csv_path = os.path.abspath(cli.csv)
    if not os.path.isfile(csv_path):
        raise SystemExit('CSV not found: {}'.format(csv_path))

    args = PhoneArgs()
    args.path_results = os.path.join(args.path_results, 'pf')
    args.csv_path = csv_path
    args.test_sequences = [os.path.splitext(os.path.basename(csv_path))[0]]
    args.estimate_mount = cli.estimate_mount
    if cli.mount_rpy is not None:
        args.mount_rpy_deg = cli.mount_rpy
        args.estimate_mount = False
    dataset = load_phone_dataset(args)
    dataset_name = dataset.datasets[0]

    iekf, torch_iekf = prepare_filter(args, dataset)
    iekf.zupt = not cli.no_zupt
    t, ang_gt, p_gt, v_gt, u = prepare_data(
        args, dataset, dataset_name, 0, to_numpy=True)
    measurements_covs = torch_iekf.forward_nets(torch.from_numpy(u).double()).detach().numpy()

    meta = load_map_metadata(cli.map)
    utm_map = is_y_down_grid(meta) and 'utm_epsg' in meta
    if utm_map:
        lla0 = PhoneDataset.load(args.path_data_save, dataset_name)['lla0']
        gps_xy, gps_yaw = utm_start_pose(lla0, meta)
        print('Start fix {:.6f}, {:.6f} -> map ({:.1f}, {:.1f}), grid convergence {:.3f} deg'.format(
            lla0[0], lla0[1], gps_xy[0], gps_xy[1], math.degrees(gps_yaw)))
    elif cli.init_x is None or cli.init_y is None:
        raise SystemExit('--init-x and --init-y are required unless the map metadata has UTM fields')
    init_xy = (
        gps_xy[0] if cli.init_x is None else cli.init_x,
        gps_xy[1] if cli.init_y is None else cli.init_y)
    if cli.yaw_offset_deg is not None:
        yaw = math.radians(cli.yaw_offset_deg)
    else:
        yaw = gps_yaw if utm_map else 0.0
    ang0, v0, p_gt_map = map_frame_pose(ang_gt, v_gt, p_gt, yaw, init_xy, cli.init_z)
    if cli.start_s > 0.0:
        k = int(np.searchsorted(t, t[0] + cli.start_s))
        if k >= t.shape[0] - 1:
            raise SystemExit('--start-s {} is past the end of the recording'.format(cli.start_s))
        init_xy = (float(p_gt_map[k, 0]), float(p_gt_map[k, 1]))
        print('Starting at sample {} (t = {:.1f} s), GPS map position ({:.1f}, {:.1f})'.format(
            k, t[k] - t[0], init_xy[0], init_xy[1]))
        t, ang_gt, v_gt, u = t[k:], ang_gt[k:], v_gt[k:], u[k:]
        measurements_covs = measurements_covs[k:]
        p_gt = p_gt[k:] - p_gt[k]
        ang0, v0, p_gt_map = map_frame_pose(ang_gt, v_gt, p_gt, yaw, init_xy, cli.init_z)

    # ------------------------------------------------------------------
    # Load map and prepare
    # ------------------------------------------------------------------
    occ = load_occupancy_map(
        cli.map, resolution=cli.res, origin_xy=(cli.origin_x, cli.origin_y),
        flipud=cli.flipud, threshold=cli.threshold)

    params = PfParams(
        num_particles=cli.num_particles,
        init_pos_sigma_m=cli.init_pos_sigma,
        init_yaw_sigma_rad=math.radians(cli.init_yaw_sigma_deg),
        blocked_penalty=cli.blocked_penalty,
        slide=not cli.no_slide,
        wall_penalty_per_s=cli.wall_penalty,
        noise_per_s=cli.noise_per_s,
        seed=cli.seed)

    n = len(t)
    dt = t[1:] - t[:-1]
    c_yaw = math.cos(yaw)
    s_yaw = math.sin(yaw)

    # ZUPT mask (computed once, uses only IMU)
    zupt_mask = None
    if iekf.zupt:
        zupt_mask = iekf.detect_zupt(t, u)
        print('ZUPT on {:.1f}% of samples'.format(100.0 * np.mean(zupt_mask)))

    # ------------------------------------------------------------------
    # Init IEKF state (sample 0)
    # ------------------------------------------------------------------
    v_mes = np.array(v_gt, dtype=np.float64, copy=True)
    v_mes[0] = v0
    Rot_all, v_all, p_all, b_omega_all, b_acc_all, Rot_c_i_all, t_c_i_all = \
        iekf.init_saved_state(dt, n, ang0)
    Rot_all[0] = iekf.from_rpy(ang0[0], ang0[1], ang0[2])
    v_all[0] = v_mes[0]
    P = iekf.init_covariance()

    # IEKF position on the map (for comparison only)
    p_kf = np.zeros((n, 3))
    p_kf[0] = [init_xy[0], init_xy[1], cli.init_z]

    # Init PF. The occupancy cloud is unchanged. --centreline uses the road graph.
    # --centreline with --no-slide moves particles freely and weights by distance to the road.
    road = None
    pf_map = occ
    if cli.centreline and cli.no_slide:
        pf_map = RoadDistanceMap(RoadGraph(cli.centreline), occ, sigma_m=cli.road_sigma)
        particles, rng, center = init_particles(pf_map, init_xy, params)
    elif cli.centreline:
        road = RoadGraph(cli.centreline)
        particles, rng = init_road_particles(road, init_xy, params)
        center = (particles[0].x, particles[0].y)
    else:
        particles, rng, center = init_particles(occ, init_xy, params)
    print('PF {} particles, init ({:.2f}, {:.2f})'.format(len(particles), center[0], center[1]))

    # Output arrays
    out_p = np.zeros((n, 2))
    out_hb = np.zeros(n)
    out_vs = np.zeros(n)
    out_conf = np.zeros(n)
    out_ess = np.zeros(n)

    if road is None:
        est = estimate_cloud(particles, pf_map, center, params)
    else:
        est = estimate_road(particles, center, params)
    out_p[0] = est['p']
    out_hb[0] = est['heading_bias']
    out_vs[0] = est['vel_scale']
    out_conf[0] = est['conf']
    out_ess[0] = est['ess']
    last_xy = est['xy']

    dt_med = float(np.median(dt)) if n > 1 else 1.0
    frame_every = max(int(round(1.0 / max(dt_med, 1e-3))), 1)
    frames = []

    def _maybe_record(i):
        if i % frame_every != 0 and i != n - 1:
            return
        frames.append([i, _particle_snapshot(particles, 400)])

    _maybe_record(0)

    # ------------------------------------------------------------------
    # Interleaved loop: 1 IEKF step + 1 PF step per sample
    # ------------------------------------------------------------------
    import time as _time
    started = _time.time()
    for i in range(1, n):
        dt_i = float(dt[i - 1])
        zupt_i = zupt_mask is not None and bool(zupt_mask[i])

        # --- IEKF step ---
        Rot_all[i], v_all[i], p_all[i], b_omega_all[i], b_acc_all[i], \
            Rot_c_i_all[i], t_c_i_all[i], P = iekf.propagate(
                Rot_all[i-1], v_all[i-1], p_all[i-1], b_omega_all[i-1],
                b_acc_all[i-1], Rot_c_i_all[i-1], t_c_i_all[i-1], P, u[i], dt_i)
        Rot_all[i], v_all[i], p_all[i], b_omega_all[i], b_acc_all[i], \
            Rot_c_i_all[i], t_c_i_all[i], P = iekf.update(
                Rot_all[i], v_all[i], p_all[i], b_omega_all[i], b_acc_all[i],
                Rot_c_i_all[i], t_c_i_all[i], P, u[i], i, measurements_covs[i], zupt_i)
        if i % iekf.n_normalize_rot == 0:
            Rot_all[i] = iekf.normalize_rot(Rot_all[i])
        if i % iekf.n_normalize_rot_c_i == 0:
            Rot_c_i_all[i] = iekf.normalize_rot(Rot_c_i_all[i])

        # IEKF velocity in map frame
        vx_map = c_yaw * v_all[i, 0] - s_yaw * v_all[i, 1]
        vy_map = s_yaw * v_all[i, 0] + c_yaw * v_all[i, 1]

        # IEKF position on the map (for comparison plot)
        p_kf[i, 0] = p_all[i, 0] + init_xy[0]
        p_kf[i, 1] = p_all[i, 1] + init_xy[1]
        p_kf[i, 2] = p_all[i, 2] + cli.init_z

        # --- PF step ---
        if road is None:
            particles, est = pf_step(particles, vx_map, vy_map, dt_i, pf_map, params, rng, last_xy)
        else:
            particles, est = pf_step_road(
                particles, vx_map, vy_map, dt_i, road, params, rng, last_xy)
        out_p[i] = est['p']
        out_hb[i] = est['heading_bias']
        out_vs[i] = est['vel_scale']
        out_conf[i] = est['conf']
        out_ess[i] = est['ess']
        last_xy = est['xy']
        _maybe_record(i)

        if i % 200 == 0 or i == n - 1:
            elapsed = _time.time() - started
            rate = i / elapsed if elapsed > 0.0 else 0.0
            remain = (n - 1 - i) / rate if rate > 0.0 else 0.0
            print('step {}/{}  {:.1f} steps/s  ETA {:.0f}s  ESS {:.0f}  conf {:.2f} m'.format(
                i, n - 1, rate, remain, est['ess'], est['conf']))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    particles_xy = np.array([[p.x, p.y] for p in particles])
    particles_log_w = np.array([p.log_w for p in particles])
    mondict = {
        't': t, 'p': out_p,
        'heading_bias': out_hb, 'vel_scale': out_vs,
        'conf': out_conf, 'ess': out_ess,
        'p_kf': p_kf, 'p_gt_map': p_gt_map,
        'particles_xy': particles_xy, 'particles_log_w': particles_log_w,
        'init_xy': np.array(init_xy), 'yaw_offset_rad': yaw,
        'map': os.path.abspath(cli.map),
    }
    if zupt_mask is not None:
        mondict['zupt'] = zupt_mask
    tag = dataset_name + '_pf'
    if cli.centreline:
        graph_name = os.path.splitext(os.path.basename(cli.centreline))[0]
        tag += '_cl' if graph_name.endswith('_centreline') else '_cl_' + graph_name.rsplit('_', 1)[-1]
    if cli.no_slide:
        tag += '_noslide'
    if cli.noise_per_s:
        tag += '_dtnoise'
    PhoneDataset.dump(mondict, args.path_results, tag + '.p')
    png = os.path.join(args.path_results, tag + '.png')
    plot_pf(
        png, occ, out_p, p_kf, p_gt_map,
        particles_xy, particles_log_w,
        '{}  {} particles  conf {:.2f} m'.format(dataset_name, cli.num_particles, out_conf[-1]))
    html = os.path.join(args.path_results, tag + '_particles.html')
    save_particle_html(
        html, occ, t, p_kf, p_gt_map, out_p, frames,
        '{}  {} particles'.format(dataset_name, cli.num_particles))
    print('Wrote {}'.format(os.path.join(args.path_results, tag + '.p')))
    print('Wrote {}'.format(png))
    print('Wrote {}'.format(html))
    print('PF final ({:.2f}, {:.2f})  conf {:.2f} m  ESS {:.0f}'.format(
        out_p[-1, 0], out_p[-1, 1], out_conf[-1], out_ess[-1]))


if __name__ == '__main__':
    main()
