"""Ballistic flight of blood droplets with gravity and quadratic air drag.

All droplets of an event are integrated together (numpy arrays), and every
step is tested as a straight segment against the surfaces, so the landing
point is exact to the segment even with a coarse time step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import GRAVITY, drag_k


@dataclass
class Launch:
    """One droplet leaving the source."""
    pos: np.ndarray           # (3,) release position, m
    vel: np.ndarray           # (3,) release velocity, m/s
    diameter: float           # m
    t_release: float = 0.0    # s after the start of the event


@dataclass
class Flight:
    landed: bool
    surface_index: int        # index into SurfaceSet.rects, -1 if it never landed
    pos: np.ndarray           # landing point (or last point)
    vel: np.ndarray           # velocity at impact
    flight_time: float
    path: np.ndarray          # (K, 3) sub-sampled trajectory for visualisation / reenactment


def accel(v, k):
    """a = g - k |v| v   (v: (N,3), k: (N,))"""
    speed = np.linalg.norm(v, axis=1, keepdims=True)
    return GRAVITY[None, :] - k[:, None] * speed * v


def fly(launches, surfaces, dt=2e-3, t_max=3.0, drag=True, path_every=10):
    """Integrate every launch until it hits a surface (or t_max). Midpoint (RK2) steps."""
    n = len(launches)
    if n == 0:
        return []
    x = np.array([l.pos for l in launches], float)
    v = np.array([l.vel for l in launches], float)
    k = drag_k([l.diameter for l in launches]) if drag else np.zeros(n)
    t = np.zeros(n)
    active = np.ones(n, bool)
    surf = np.full(n, -1, int)
    vimp = v.copy()
    paths = [[x[i].copy()] for i in range(n)]
    steps = int(np.ceil(t_max / dt))
    for step in range(steps):
        a_idx = np.flatnonzero(active)
        if a_idx.size == 0:
            break
        xa, va, ka = x[a_idx], v[a_idx], k[a_idx]
        vm = va + 0.5 * dt * accel(va, ka)
        x_new = xa + dt * vm
        v_new = va + dt * accel(vm, ka)
        hit_idx, frac, pts = surfaces.first_hits(xa, x_new)
        hit = hit_idx >= 0
        # landed droplets
        if hit.any():
            h = a_idx[hit]
            f = frac[hit][:, None]
            x[h] = pts[hit]
            vimp[h] = va[hit] + f * (v_new[hit] - va[hit])
            t[h] += frac[hit] * dt
            surf[h] = hit_idx[hit]
            active[h] = False
            for i in h:
                paths[i].append(x[i].copy())
        # still flying
        fl = a_idx[~hit]
        x[fl] = x_new[~hit]
        v[fl] = v_new[~hit]
        t[fl] += dt
        if step % path_every == 0:
            for i in fl:
                paths[i].append(x[i].copy())
    out = []
    for i in range(n):
        landed = surf[i] >= 0
        out.append(Flight(bool(landed), int(surf[i]), x[i].copy(),
                          (vimp[i] if landed else v[i]).copy(), float(t[i]), np.array(paths[i])))
    return out
