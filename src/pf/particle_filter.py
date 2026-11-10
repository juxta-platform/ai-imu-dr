"""Lightweight particle filter driven by IEKF velocity output.

Stage 1 (caller): run the IEKF once on the raw IMU to get a velocity time series.
Stage 2 (this module): each particle holds (x, y, heading_bias, heading_rate,
vel_scale, blocked_ema, log_w).  A step rotates the IEKF velocity by the
particle's heading_bias, scales it by vel_scale, integrates position, and the
occupancy map reweights.

Map array: 0 = free, 1 = occupied. Axis 0 is x and axis 1 is y, in metres from
the grid origin. A sibling ``*_metadata.json`` may supply ``grid_cell_size_m``;
those maps are flipped vertically, same as the RoNIN floorplan loader.
"""
import json
import math
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from utils_numpy_filter import NUMPYIEKF


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class PfParams(object):
    def __init__(self, num_particles=1000, init_pos_sigma_m=1.5, init_yaw_sigma_rad=0.20,
                 heading_walk_sigma_rad=0.004,
                 init_heading_rate_sigma_radps=0.012,
                 heading_rate_walk_sigma_radps=0.0004,
                 heading_rate_bound_radps=0.035,
                 init_scale_mean=1.0, init_scale_sigma=0.15,
                 scale_walk_sigma=0.012, scale_bounds=(0.5, 2.0),
                 vel_noise_frac=0.25, vel_noise_floor_mps=0.02,
                 pos_noise_sigma_m=0.04,
                 blocked_penalty=3.0, blocked_ema_alpha=0.15,
                 state_reversion_per_s=0.04,
                 resample_ess_frac=0.25,
                 cluster_radius_m=2.5, cluster_switch_ratio=2.5,
                 roughen_pos_sigma_m=0.03, roughen_heading_sigma_rad=0.004,
                 slide=True, wall_penalty_per_s=10.0, noise_per_s=False,
                 seed=0x5EED1234):
        self.num_particles = int(num_particles)
        self.init_pos_sigma_m = float(init_pos_sigma_m)
        self.init_yaw_sigma_rad = float(init_yaw_sigma_rad)
        self.heading_walk_sigma_rad = float(heading_walk_sigma_rad)
        self.init_heading_rate_sigma_radps = float(init_heading_rate_sigma_radps)
        self.heading_rate_walk_sigma_radps = float(heading_rate_walk_sigma_radps)
        self.heading_rate_bound_radps = float(heading_rate_bound_radps)
        self.init_scale_mean = float(init_scale_mean)
        self.init_scale_sigma = float(init_scale_sigma)
        self.scale_walk_sigma = float(scale_walk_sigma)
        self.scale_bounds = (float(scale_bounds[0]), float(scale_bounds[1]))
        self.vel_noise_frac = float(vel_noise_frac)
        self.vel_noise_floor_mps = float(vel_noise_floor_mps)
        self.pos_noise_sigma_m = float(pos_noise_sigma_m)
        self.blocked_penalty = float(blocked_penalty)
        self.blocked_ema_alpha = float(blocked_ema_alpha)
        self.state_reversion_per_s = float(state_reversion_per_s)
        self.resample_ess_frac = float(resample_ess_frac)
        self.cluster_radius_m = float(cluster_radius_m)
        self.cluster_switch_ratio = float(cluster_switch_ratio)
        self.roughen_pos_sigma_m = float(roughen_pos_sigma_m)
        self.roughen_heading_sigma_rad = float(roughen_heading_sigma_rad)
        self.slide = bool(slide)
        self.wall_penalty_per_s = float(wall_penalty_per_s)
        self.noise_per_s = bool(noise_per_s)
        self.seed = int(seed)


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------

class Particle(object):
    __slots__ = ('x', 'y', 'heading_bias', 'heading_rate', 'vel_scale',
                 'blocked_ema', 'log_w')

    def __init__(self, x, y, heading_bias, heading_rate, vel_scale,
                 blocked_ema=0.0, log_w=0.0):
        self.x = float(x)
        self.y = float(y)
        self.heading_bias = float(heading_bias)
        self.heading_rate = float(heading_rate)
        self.vel_scale = float(vel_scale)
        self.blocked_ema = float(blocked_ema)
        self.log_w = float(log_w)

    def copy(self):
        return Particle(self.x, self.y, self.heading_bias, self.heading_rate,
                        self.vel_scale, self.blocked_ema, self.log_w)


# ---------------------------------------------------------------------------
# Occupancy map (unchanged)
# ---------------------------------------------------------------------------

class OccupancyMap(object):
    """Boolean occupancy indexed ``[ix, iy]`` with ix along x and iy along y."""

    def __init__(self, occupied, resolution, origin_x=0.0, origin_y=0.0):
        self.occupied = np.ascontiguousarray(occupied, dtype=bool)
        if self.occupied.ndim != 2:
            raise ValueError('occupancy grid must be 2D')
        self.res = float(resolution)
        if self.res <= 0.0:
            raise ValueError('resolution must be positive')
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.nx, self.ny = self.occupied.shape

    def contains_free(self, x, y):
        ix = math.floor((x - self.origin_x) / self.res)
        iy = math.floor((y - self.origin_y) / self.res)
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return False
        return not bool(self.occupied[ix, iy])

    def cost(self, x, y):
        """Per-second log-weight cost factor at (x, y), used with ``slide=False``."""
        return 0.0 if self.contains_free(x, y) else 1.0

    def nearest_free_point(self, x, y):
        if self.contains_free(x, y):
            return (float(x), float(y))
        max_x = self.origin_x + self.nx * self.res - 1e-9
        max_y = self.origin_y + self.ny * self.res - 1e-9
        xc = min(max(float(x), self.origin_x), max_x)
        yc = min(max(float(y), self.origin_y), max_y)
        ix = int(math.floor((xc - self.origin_x) / self.res))
        iy = int(math.floor((yc - self.origin_y) / self.res))
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        q = deque([(ix, iy)])
        seen = {(ix, iy)}
        while q:
            cx, cy = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in seen:
                    continue
                if nx < 0 or ny < 0 or nx >= self.nx or ny >= self.ny:
                    continue
                if not self.occupied[nx, ny]:
                    return (
                        self.origin_x + (nx + 0.5) * self.res,
                        self.origin_y + (ny + 0.5) * self.res)
                seen.add((nx, ny))
                q.append((nx, ny))
        raise ValueError('occupancy map has no free cells')

    def cast_ray(self, x, y, dx, dy):
        length = math.hypot(dx, dy)
        if length < 1e-12:
            return (x, y, 1.0)
        if not self.contains_free(x, y):
            return (x + dx, y + dy, 1.0)
        step = self.res * 0.4
        n = max(int(math.ceil(length / step)), 1)
        reached = 0.0
        for i in range(1, n + 1):
            t = i / n
            if self.contains_free(x + dx * t, y + dy * t):
                reached = t
            else:
                break
        return (x + dx * reached, y + dy * reached, reached)

    def move_with_slide(self, x, y, dx, dy):
        """Stop at a wall. Axis-aligned slides are kept when they travel farther."""
        want = math.hypot(dx, dy)
        fx, fy, frac = self.cast_ray(x, y, dx, dy)
        if frac >= 1.0:
            return (fx, fy, want)
        direct = (fx, fy, frac * want)

        xa, ya, fa = self.cast_ray(x, y, dx, 0.0)
        xa2, ya2, fa2 = self.cast_ray(xa, ya, 0.0, dy)
        slide_xy = (xa2, ya2, fa * abs(dx) + fa2 * abs(dy))

        xb, yb, fb = self.cast_ray(x, y, 0.0, dy)
        xb2, yb2, fb2 = self.cast_ray(xb, yb, dx, 0.0)
        slide_yx = (xb2, yb2, fb * abs(dy) + fb2 * abs(dx))
        return max(direct, slide_xy, slide_yx, key=lambda item: item[2])


# ---------------------------------------------------------------------------
# Map / metadata loading (unchanged)
# ---------------------------------------------------------------------------

def load_map_metadata(path):
    """Sibling ``*_metadata.json`` of an occupancy ``.npy``, or None."""
    path = os.path.abspath(path)
    meta_path = path[:-4] + '_metadata.json' if path.endswith('.npy') else ''
    if not meta_path or not os.path.isfile(meta_path):
        return None
    with open(meta_path, 'r') as handle:
        return json.load(handle)


def is_y_down_grid(meta):
    """Grid stored ``[row, col]`` with rows along local y pointing south (OSM exports)."""
    return meta is not None and meta.get('y_direction') == 'down' and 'grid_rows' in meta


def load_occupancy_map(path, resolution=None, origin_xy=(0.0, 0.0), flipud=None, threshold=0.5):
    """Load an occupancy ``.npy``.

    ``flipud=None`` flips when a metadata json sits beside the file (RoNIN
    floorplans) and otherwise leaves the array as stored. A ``y_direction: down``
    grid is converted to x east / y north with its origin from the metadata.
    """
    path = os.path.abspath(path)
    grid = np.load(path)
    grid = np.squeeze(grid)
    if grid.ndim == 3 and grid.shape[-1] <= 4:
        grid = grid[..., 0]
    if grid.ndim != 2:
        raise ValueError('occupancy npy must be 2D, got shape {}'.format(grid.shape))

    meta = load_map_metadata(path)
    meta_path = path[:-4] + '_metadata.json'
    has_meta = meta is not None
    if has_meta and resolution is None:
        resolution = meta.get('grid_cell_size_m', meta.get('cell_size_m'))
    if is_y_down_grid(meta):
        if grid.shape != (int(meta['grid_rows']), int(meta['grid_cols'])):
            raise ValueError('grid shape {} does not match metadata rows/cols'.format(grid.shape))
        grid = np.flipud(grid).T
        origin_xy = (float(meta['grid_min_x']), -float(meta['grid_max_y']))
        do_flip = True
    else:
        do_flip = has_meta if flipud is None else bool(flipud)
        if do_flip:
            grid = np.flipud(grid)
    if resolution is None:
        raise ValueError(
            'pass --res, or provide grid_cell_size_m in {}'.format(meta_path or path))

    if np.issubdtype(grid.dtype, np.bool_):
        occupied = grid
    else:
        vals = grid.astype(np.float64)
        finite = np.isfinite(vals)
        if np.any(finite) and np.max(vals[finite]) > 1.5:
            vals = vals / 255.0
        occupied = ~np.isfinite(vals) | (vals > threshold)
    if not np.any(~occupied):
        raise ValueError('occupancy map has no free cells')

    print('Map {}  shape {}  res {} m  free {:.1f}%  origin ({}, {})  flipud {}'.format(
        os.path.basename(path), occupied.shape, float(resolution),
        100.0 * np.mean(~occupied), origin_xy[0], origin_xy[1], do_flip))
    return OccupancyMap(occupied, float(resolution), origin_xy[0], origin_xy[1])


# ---------------------------------------------------------------------------
# Weight utilities
# ---------------------------------------------------------------------------

def normalized_weights(log_w):
    log_w = np.asarray(log_w, dtype=np.float64)
    n = log_w.shape[0]
    peak = np.max(log_w)
    if not np.isfinite(peak):
        return np.full(n, 1.0 / n)
    weights = np.exp(log_w - peak)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0.0:
        return np.full(n, 1.0 / n)
    return weights / total


def _normal(rng, sigma):
    if sigma <= 0.0:
        return 0.0
    return float(rng.normal(0.0, sigma))


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def init_particles(occ, init_xy, params):
    """Spawn particles around the start position."""
    if params.num_particles < 1:
        raise ValueError('num_particles must be positive')
    cx, cy = occ.nearest_free_point(init_xy[0], init_xy[1])
    rng = np.random.default_rng(params.seed)
    particles = []
    for _ in range(params.num_particles):
        x, y = cx, cy
        if params.init_pos_sigma_m > 0.0:
            for _try in range(30):
                tx = cx + _normal(rng, params.init_pos_sigma_m)
                ty = cy + _normal(rng, params.init_pos_sigma_m)
                if occ.contains_free(tx, ty):
                    x, y = tx, ty
                    break
        vel_scale = min(max(
            params.init_scale_mean + _normal(rng, params.init_scale_sigma),
            params.scale_bounds[0]), params.scale_bounds[1])
        particles.append(Particle(
            x=x, y=y,
            heading_bias=_normal(rng, params.init_yaw_sigma_rad),
            heading_rate=_normal(rng, params.init_heading_rate_sigma_radps),
            vel_scale=vel_scale))
    return particles, rng, (cx, cy)


# ---------------------------------------------------------------------------
# Propagate one particle
# ---------------------------------------------------------------------------

def step_particle(p, dt, vx, vy, occ, params, rng, mean_scale, mean_rate):
    """Rotate IEKF velocity by heading_bias, scale, integrate, map check."""
    prm = params
    revert = prm.state_reversion_per_s * dt
    k = math.sqrt(dt) if prm.noise_per_s else 1.0

    # evolve heading rate
    p.heading_rate = min(max(
        p.heading_rate
        + revert * (mean_rate - p.heading_rate)
        + _normal(rng, k * prm.heading_rate_walk_sigma_radps),
        -prm.heading_rate_bound_radps), prm.heading_rate_bound_radps)
    p.heading_bias += p.heading_rate * dt + _normal(rng, k * prm.heading_walk_sigma_rad)

    # evolve vel scale
    p.vel_scale = min(max(
        p.vel_scale
        + revert * (mean_scale - p.vel_scale)
        + _normal(rng, k * prm.scale_walk_sigma),
        prm.scale_bounds[0]), prm.scale_bounds[1])

    # rotate and scale the IEKF velocity
    speed = math.hypot(vx, vy)
    vel_sigma = prm.vel_noise_frac * speed + prm.vel_noise_floor_mps
    pos_sigma = k * prm.pos_noise_sigma_m * min(speed / 1.0, 1.0)
    sin_b = math.sin(p.heading_bias)
    cos_b = math.cos(p.heading_bias)
    rvx = (cos_b * vx - sin_b * vy) * p.vel_scale + _normal(rng, vel_sigma)
    rvy = (sin_b * vx + cos_b * vy) * p.vel_scale + _normal(rng, vel_sigma)
    dx = rvx * dt + _normal(rng, pos_sigma)
    dy = rvy * dt + _normal(rng, pos_sigma)

    # map constraint
    if not prm.slide:
        p.x += dx
        p.y += dy
        blocked = occ.cost(p.x, p.y)
        p.log_w -= prm.wall_penalty_per_s * blocked * dt
        p.blocked_ema += prm.blocked_ema_alpha * (min(blocked, 1.0) - p.blocked_ema)
        return
    nx, ny, _ = occ.move_with_slide(p.x, p.y, dx, dy)
    want = math.hypot(dx, dy)
    blocked_frac = 0.0
    if want > 1e-9:
        progress = ((nx - p.x) * dx + (ny - p.y) * dy) / want
        blocked_frac = min(max(1.0 - progress / want, 0.0), 1.0)
    p.log_w -= prm.blocked_penalty * blocked_frac
    p.blocked_ema += prm.blocked_ema_alpha * (blocked_frac - p.blocked_ema)
    p.x = nx
    p.y = ny


# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------

def _cluster(positions, weights, seed_x, seed_y, radius):
    d2 = (positions[:, 0] - seed_x) ** 2 + (positions[:, 1] - seed_y) ** 2
    mask = d2 <= radius * radius
    if not np.any(mask):
        return {'p': None, 'mass': 0.0, 'index': None}
    chosen = weights[mask]
    mass = float(chosen.sum())
    chosen = chosen / mass
    idx = np.flatnonzero(mask)
    mode = int(idx[np.argmax(weights[idx])])
    return {
        'p': chosen.dot(positions[mask]),
        'mass': mass,
        'index': mode,
    }


def estimate_cloud(particles, occ, last_xy, params):
    """Cluster mean position."""
    n = len(particles)
    positions = np.empty((n, 2))
    log_w = np.empty(n)
    for k, part in enumerate(particles):
        positions[k, 0] = part.x
        positions[k, 1] = part.y
        log_w[k] = part.log_w
    weights = normalized_weights(log_w)
    ess = 1.0 / float(np.dot(weights, weights))
    mean_xy = weights.dot(positions)
    spread = np.sum((positions - mean_xy) ** 2, axis=1)
    conf = math.sqrt(float(weights.dot(spread)))

    best = int(np.argmax(weights))
    top = _cluster(positions, weights,
                   positions[best, 0], positions[best, 1], params.cluster_radius_m)
    if last_xy is None:
        chosen = top
    else:
        prev = _cluster(positions, weights, last_xy[0], last_xy[1], params.cluster_radius_m)
        if prev['mass'] * params.cluster_switch_ratio >= top['mass']:
            chosen = prev
        else:
            chosen = top

    p_mean = np.array(chosen['p'], dtype=np.float64, copy=True)
    if not occ.contains_free(p_mean[0], p_mean[1]):
        sx, sy = occ.nearest_free_point(p_mean[0], p_mean[1])
        p_mean[0] = sx
        p_mean[1] = sy
    mode = particles[chosen['index']]
    return {
        'p': p_mean,
        'heading_bias': mode.heading_bias,
        'vel_scale': mode.vel_scale,
        'conf': conf,
        'ess': ess,
        'xy': (float(chosen['p'][0]), float(chosen['p'][1])),
    }


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------

def maybe_resample(particles, rng, occ, params):
    n = len(particles)
    log_w = np.array([p.log_w for p in particles], dtype=np.float64)
    weights = normalized_weights(log_w)
    ess = 1.0 / float(np.dot(weights, weights))
    if ess >= params.resample_ess_frac * n:
        return particles

    positions = (np.arange(n) + float(rng.random())) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    indices = np.clip(np.searchsorted(cdf, positions, side='left'), 0, n - 1)
    children = []
    for idx in indices:
        parent = particles[int(idx)]
        child = parent.copy()
        child.log_w = 0.0
        if params.roughen_pos_sigma_m > 0.0:
            child.x += _normal(rng, params.roughen_pos_sigma_m)
            child.y += _normal(rng, params.roughen_pos_sigma_m)
            if not occ.contains_free(child.x, child.y):
                child.x = parent.x
                child.y = parent.y
        if params.roughen_heading_sigma_rad > 0.0:
            child.heading_bias += _normal(rng, params.roughen_heading_sigma_rad)
        children.append(child)
    return children


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_estimate(out, i, est):
    out['p'][i, 0] = est['p'][0]
    out['p'][i, 1] = est['p'][1]
    out['heading_bias'][i] = est['heading_bias']
    out['vel_scale'][i] = est['vel_scale']
    out['conf'][i] = est['conf']
    out['ess'][i] = est['ess']


def _particle_snapshot(particles, limit):
    """Positions and states for one animation frame."""
    n = len(particles)
    log_w = np.empty(n)
    for k, part in enumerate(particles):
        log_w[k] = part.log_w
    weights = normalized_weights(log_w)
    if n > limit:
        index = np.linspace(0, n - 1, limit, dtype=int)
    else:
        index = np.arange(n)
    snapshot = []
    for i in index:
        part = particles[int(i)]
        snapshot.append([
            part.x, part.y,
            math.degrees(part.heading_bias),
            part.vel_scale,
            part.blocked_ema,
            part.log_w,
            float(weights[int(i)]),
        ])
    return snapshot


# ---------------------------------------------------------------------------
# One PF step (called from the interleaved loop in main_pf.py)
# ---------------------------------------------------------------------------

def pf_step(particles, vx, vy, dt, occ, params, rng, last_xy):
    """One PF step: propagate all particles, estimate, resample.

    Returns (particles, est) where est is the estimate dict.
    """
    mean_scale = sum(p.vel_scale for p in particles) / len(particles)
    mean_rate = sum(p.heading_rate for p in particles) / len(particles)

    for part in particles:
        step_particle(part, dt, vx, vy, occ, params, rng, mean_scale, mean_rate)

    est = estimate_cloud(particles, occ, last_xy, params)
    particles = maybe_resample(particles, rng, occ, params)
    return particles, est
