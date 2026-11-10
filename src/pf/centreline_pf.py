"""Particle filter on a centreline graph.

Same IEKF velocity input as the occupancy filter. A particle stores an edge,
a distance along that edge, and a direction. Heading is the edge tangent.
At a node the particle is sent down one of the leaving edges, with the
probability set by how well that edge matches the inertial heading.
"""
import json
import math

import numpy as np

from scipy.ndimage import distance_transform_edt

from particle_filter import OccupancyMap, PfParams, normalized_weights


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _normal(rng, sigma):
    if sigma <= 0.0:
        return 0.0
    return float(rng.normal(0.0, sigma))


class RoadGraph(object):
    def __init__(self, path):
        with open(path) as handle:
            raw = json.load(handle)
        self.pts = []
        self.cum = []
        self.length = []
        self.n0 = []
        self.n1 = []
        self.adj = {}
        for edge in raw['edges']:
            pts = np.asarray(edge['pts'], dtype=np.float64)
            if pts.shape[0] < 2:
                continue
            step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(step)])
            length = float(cum[-1])
            if length < 1e-3:
                continue
            idx = len(self.pts)
            self.pts.append(pts)
            self.cum.append(cum)
            self.length.append(length)
            n0 = int(edge['n0'])
            n1 = int(edge['n1'])
            self.n0.append(n0)
            self.n1.append(n1)
            self.adj.setdefault(n0, []).append((idx, 1))
            self.adj.setdefault(n1, []).append((idx, -1))
        self.length = np.asarray(self.length, dtype=np.float64)
        self.n0 = np.asarray(self.n0, dtype=np.int64)
        self.n1 = np.asarray(self.n1, dtype=np.int64)
        print('Centreline {} edges  {:.0f} m'.format(len(self.pts), float(self.length.sum())))

    def pose(self, edge, s):
        """Point and +s unit tangent at distance s from the start of the edge."""
        pts = self.pts[edge]
        cum = self.cum[edge]
        s = min(max(float(s), 0.0), float(cum[-1]))
        i = int(np.searchsorted(cum, s, side='right') - 1)
        i = min(max(i, 0), len(pts) - 2)
        span = float(cum[i + 1] - cum[i])
        t = 0.0 if span < 1e-9 else (s - float(cum[i])) / span
        x = float(pts[i, 0] + t * (pts[i + 1, 0] - pts[i, 0]))
        y = float(pts[i, 1] + t * (pts[i + 1, 1] - pts[i, 1]))
        tx = float(pts[i + 1, 0] - pts[i, 0])
        ty = float(pts[i + 1, 1] - pts[i, 1])
        nrm = math.hypot(tx, ty) or 1.0
        return x, y, tx / nrm, ty / nrm

    def nearest(self, x, y, radius):
        """Edges whose polyline comes within ``radius`` of (x, y).

        Returns a list of (edge, s, distance).
        """
        hits = []
        r2 = radius * radius
        for e, pts in enumerate(self.pts):
            if (pts[:, 0].min() - radius > x or pts[:, 0].max() + radius < x or
                    pts[:, 1].min() - radius > y or pts[:, 1].max() + radius < y):
                continue
            best_d2 = r2
            best_s = 0.0
            cum = self.cum[e]
            for i in range(len(pts) - 1):
                ax, ay = float(pts[i, 0]), float(pts[i, 1])
                bx, by = float(pts[i + 1, 0]), float(pts[i + 1, 1])
                dx, dy = bx - ax, by - ay
                span2 = dx * dx + dy * dy
                t = 0.0 if span2 < 1e-12 else ((x - ax) * dx + (y - ay) * dy) / span2
                t = min(max(t, 0.0), 1.0)
                px, py = ax + t * dx, ay + t * dy
                d2 = (px - x) ** 2 + (py - y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_s = float(cum[i]) + t * math.sqrt(span2)
            if best_d2 < r2:
                hits.append((e, best_s, math.sqrt(best_d2)))
        return hits


class RoadDistanceMap(OccupancyMap):
    """Free-moving particles weighted by distance to the nearest centreline.

    Rasterised over the occupancy map extent. Cells farther than ``3 * sigma_m``
    from a road count as occupied; ``cost`` is ``0.5 * (d / sigma_m)^2``.
    """

    def __init__(self, graph, occ, sigma_m=8.0, res=1.0):
        nx = int(math.ceil(occ.nx * occ.res / res))
        ny = int(math.ceil(occ.ny * occ.res / res))
        on_road = np.zeros((nx, ny), dtype=bool)
        for pts in graph.pts:
            for a, b in zip(pts[:-1], pts[1:]):
                n = max(int(math.ceil(np.linalg.norm(b - a) / (0.5 * res))), 1)
                seg = a + np.linspace(0.0, 1.0, n + 1)[:, None] * (b - a)
                ix = np.floor((seg[:, 0] - occ.origin_x) / res).astype(int)
                iy = np.floor((seg[:, 1] - occ.origin_y) / res).astype(int)
                ok = (ix >= 0) & (iy >= 0) & (ix < nx) & (iy < ny)
                on_road[ix[ok], iy[ok]] = True
        self.dist = distance_transform_edt(~on_road, sampling=res).astype(np.float32)
        self.sigma_m = float(sigma_m)
        OccupancyMap.__init__(self, self.dist > 3.0 * self.sigma_m, res, occ.origin_x, occ.origin_y)
        print('Road distance field {}x{} at {} m  sigma {} m'.format(nx, ny, res, sigma_m))

    def cost(self, x, y):
        ix = math.floor((x - self.origin_x) / self.res)
        iy = math.floor((y - self.origin_y) / self.res)
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return 0.5 * 9.0
        d = float(self.dist[ix, iy]) / self.sigma_m
        return 0.5 * d * d


class RoadParticle(object):
    __slots__ = ('x', 'y', 'heading_bias', 'heading_rate', 'vel_scale',
                 'blocked_ema', 'log_w', 'edge', 's', 'direction')

    def __init__(self, edge, s, direction, heading_bias, heading_rate, vel_scale):
        self.edge = int(edge)
        self.s = float(s)
        self.direction = 1 if direction >= 0 else -1
        self.heading_bias = float(heading_bias)
        self.heading_rate = float(heading_rate)
        self.vel_scale = float(vel_scale)
        self.blocked_ema = 0.0
        self.log_w = 0.0
        self.x = 0.0
        self.y = 0.0

    def copy(self):
        other = RoadParticle(
            self.edge, self.s, self.direction, self.heading_bias, self.heading_rate, self.vel_scale)
        other.blocked_ema = self.blocked_ema
        other.log_w = self.log_w
        other.x = self.x
        other.y = self.y
        return other

    def place(self, graph):
        tx, ty = 0.0, 0.0
        self.x, self.y, tx, ty = graph.pose(self.edge, self.s)
        if self.direction < 0:
            tx, ty = -tx, -ty
        return tx, ty


def init_road_particles(graph, init_xy, params, radius=40.0):
    if params.num_particles < 1:
        raise ValueError('num_particles must be positive')
    hits = graph.nearest(init_xy[0], init_xy[1], radius)
    if not hits:
        hits = graph.nearest(init_xy[0], init_xy[1], radius * 4.0)
    if not hits:
        raise ValueError('no centreline within {:.0f} m of the start'.format(radius * 4.0))
    dist = np.array([h[2] for h in hits])
    # Wide enough that the next street over still gets particles, not only the nearest segment.
    pick_w = np.exp(-0.5 * (dist / 15.0) ** 2)
    pick_w /= pick_w.sum()
    rng = np.random.default_rng(params.seed)
    particles = []
    for _ in range(params.num_particles):
        e, s, _d = hits[int(rng.choice(len(hits), p=pick_w))]
        s = min(max(s + _normal(rng, params.init_pos_sigma_m), 0.0), float(graph.length[e]))
        direction = 1 if rng.random() < 0.5 else -1
        vel_scale = min(max(
            params.init_scale_mean + _normal(rng, params.init_scale_sigma),
            params.scale_bounds[0]), params.scale_bounds[1])
        part = RoadParticle(
            e, s, direction,
            _normal(rng, params.init_yaw_sigma_rad),
            _normal(rng, params.init_heading_rate_sigma_radps),
            vel_scale)
        part.place(graph)
        particles.append(part)
    return particles, rng


def _choose_exit(graph, node, arrival_edge, heading, rng, sigma=0.5):
    """Pick a leaving edge. ``heading`` is the inertial heading, radians."""
    options = graph.adj.get(int(node), [])
    if not options:
        return None
    logs = []
    for edge, direction in options:
        _x, _y, tx, ty = graph.pose(edge, 0.0 if direction > 0 else float(graph.length[edge]))
        if direction < 0:
            tx, ty = -tx, -ty
        err = _wrap(heading - math.atan2(ty, tx))
        logp = -0.5 * (err / sigma) ** 2
        if edge == arrival_edge:
            logp -= 4.0
        logs.append(logp)
    logs = np.asarray(logs, dtype=np.float64)
    logs -= logs.max()
    prob = np.exp(logs)
    prob /= prob.sum()
    return options[int(rng.choice(len(options), p=prob))]


def step_road_particle(part, dt, vx, vy, graph, params, rng, mean_scale, mean_rate):
    prm = params
    revert = prm.state_reversion_per_s * dt
    k = math.sqrt(dt) if prm.noise_per_s else 1.0
    part.heading_rate = min(max(
        part.heading_rate
        + revert * (mean_rate - part.heading_rate)
        + _normal(rng, k * prm.heading_rate_walk_sigma_radps),
        -prm.heading_rate_bound_radps), prm.heading_rate_bound_radps)
    part.heading_bias += part.heading_rate * dt + _normal(rng, k * prm.heading_walk_sigma_rad)
    part.vel_scale = min(max(
        part.vel_scale
        + revert * (mean_scale - part.vel_scale)
        + _normal(rng, k * prm.scale_walk_sigma),
        prm.scale_bounds[0]), prm.scale_bounds[1])

    sin_b = math.sin(part.heading_bias)
    cos_b = math.cos(part.heading_bias)
    rvx = cos_b * vx - sin_b * vy
    rvy = sin_b * vx + cos_b * vy
    speed = math.hypot(rvx, rvy)
    if speed < 1e-3:
        part.place(graph)
        part.blocked_ema += prm.blocked_ema_alpha * (0.0 - part.blocked_ema)
        return

    heading = math.atan2(rvy, rvx)
    tx, ty = part.place(graph)
    err_fwd = abs(_wrap(heading - math.atan2(ty, tx)))
    err_back = abs(_wrap(heading - math.atan2(-ty, -tx)))
    # Hysteresis so a heading across the road does not flip the particle every sample.
    if err_back + 0.35 < err_fwd:
        part.direction = -part.direction
        tx, ty = -tx, -ty
        part.log_w -= 1.0
    err = _wrap(heading - math.atan2(ty, tx))
    part.log_w -= 0.4 * err * err * dt
    part.blocked_ema += prm.blocked_ema_alpha * (abs(err) / math.pi - part.blocked_ema)
    # Travel the inertial distance along the road. The heading error is a weight, not a brake,
    # so a particle still reaches the next junction when the road it is on points the wrong way.
    ds = speed * part.vel_scale * dt

    hops = 0
    while ds > 1e-6 and hops < 8:
        hops += 1
        if part.direction > 0:
            room = float(graph.length[part.edge]) - part.s
            if ds <= room + 1e-9:
                part.s += ds
                ds = 0.0
                break
            ds -= room
            node = int(graph.n1[part.edge])
        else:
            room = part.s
            if ds <= room + 1e-9:
                part.s -= ds
                ds = 0.0
                break
            ds -= room
            node = int(graph.n0[part.edge])
        nxt = _choose_exit(graph, node, part.edge, heading, rng)
        if nxt is None:
            part.direction = -part.direction
            break
        part.edge, part.direction = nxt
        part.s = 0.0 if part.direction > 0 else float(graph.length[part.edge])
    part.place(graph)


def estimate_road(particles, last_xy, params):
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
    radius = params.cluster_radius_m

    def cluster(seed_x, seed_y):
        d2 = (positions[:, 0] - seed_x) ** 2 + (positions[:, 1] - seed_y) ** 2
        mask = d2 <= radius * radius
        if not np.any(mask):
            return None
        mass = float(weights[mask].sum())
        chosen = weights[mask] / mass
        idx = np.flatnonzero(mask)
        mode = int(idx[np.argmax(weights[idx])])
        return {'p': chosen.dot(positions[mask]), 'mass': mass, 'index': mode}

    top = cluster(positions[best, 0], positions[best, 1])
    if last_xy is None:
        chosen = top
    else:
        prev = cluster(last_xy[0], last_xy[1])
        if prev is not None and prev['mass'] * params.cluster_switch_ratio >= top['mass']:
            chosen = prev
        else:
            chosen = top
    mode = particles[chosen['index']]
    return {
        'p': np.array(chosen['p'], dtype=np.float64, copy=True),
        'heading_bias': mode.heading_bias,
        'vel_scale': mode.vel_scale,
        'conf': conf,
        'ess': ess,
        'xy': (float(chosen['p'][0]), float(chosen['p'][1])),
    }


def maybe_resample_road(particles, rng, graph, params):
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
            child.s += _normal(rng, params.roughen_pos_sigma_m) * child.direction
            child.s = min(max(child.s, 0.0), float(graph.length[child.edge]))
        if params.roughen_heading_sigma_rad > 0.0:
            child.heading_bias += _normal(rng, params.roughen_heading_sigma_rad)
        child.place(graph)
        children.append(child)
    return children


def pf_step_road(particles, vx, vy, dt, graph, params, rng, last_xy):
    mean_scale = sum(p.vel_scale for p in particles) / len(particles)
    mean_rate = sum(p.heading_rate for p in particles) / len(particles)
    for part in particles:
        step_road_particle(part, dt, vx, vy, graph, params, rng, mean_scale, mean_rate)
    est = estimate_road(particles, last_xy, params)
    particles = maybe_resample_road(particles, rng, graph, params)
    return particles, est
