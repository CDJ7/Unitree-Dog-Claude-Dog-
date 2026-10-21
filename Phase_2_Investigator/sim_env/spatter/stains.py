"""Turn a landed droplet into a stain: an ellipse, a tail and satellite drops.

Model (standard BPA geometry):
  impact angle alpha: sin(alpha) = |v . n| / |v|   (90 deg = straight on)
  stain width  W = d * spread(v_normal)             (heuristic spread factor)
  stain length L = W / sin(alpha)                   (so sin(alpha) = W / L)
  the long axis points along the travel direction projected on the surface,
  the tail and satellites sit on the far (downstream) side.

The spread factor is a simple heuristic, not a fluid-dynamics model. It only
sets how big stains are; the angle and direction, which the analysis uses,
follow the geometry exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

MIN_ALPHA_DEG = 6.0           # below this a droplet skims and leaves a streak; clamp L


def spread_factor(v_normal):
    """Stain width / droplet diameter as a function of the normal impact speed (m/s)."""
    return float(np.clip(1.2 + 0.45 * np.sqrt(max(v_normal, 0.0)), 1.2, 4.0))


@dataclass
class Stain:
    id: str
    surface_id: str
    event: int
    droplet_id: str | None    # None for smears
    kind: str                 # "parent" | "satellite"
    center_uv: list           # ellipse centre in the surface (u, v) frame, m
    center_world: list
    width: float              # m (minor axis, full)
    length: float             # m (major axis, full, tail excluded)
    impact_angle_deg: float
    gamma_deg: float          # travel direction in the surface frame, deg CCW from +u
    dir_world: list           # unit travel direction projected on the surface
    tail_length: float = 0.0  # m beyond the downstream end of the ellipse
    parent_id: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Smear:
    """A transfer / wipe pattern: a wide band dragged across a surface."""
    id: str
    surface_id: str
    event: int
    path_uv: list             # (K, 2) centre line, start -> end
    width: float              # m
    feather_frac: float = 0.3 # last fraction of the path that breaks into streaks
    path_world: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def impact_geometry(vel, normal_face):
    """(alpha_deg, travel dir on the surface, unit) from impact velocity and the face normal.

    normal_face must point toward the side the droplet came from (opposite to vel).
    """
    v = np.asarray(vel, float)
    n = np.asarray(normal_face, float)
    speed = np.linalg.norm(v)
    vn = -(v @ n)
    sin_a = np.clip(vn / speed, 0.0, 1.0)
    alpha = np.degrees(np.arcsin(sin_a))
    vt = v + vn * n
    nt = np.linalg.norm(vt)
    d = vt / nt if nt > 1e-9 else np.zeros(3)
    return float(alpha), d, float(vn)


def make_stain(sid, rect, event, droplet_id, hit_point, vel, diameter, rng, kind="parent", parent_id=None):
    """Stain left by one droplet that hit `rect` at `hit_point` with velocity `vel`."""
    n_face = rect.normal if (np.asarray(vel) @ rect.normal) < 0 else -rect.normal
    alpha, d3, vn = impact_geometry(vel, n_face)
    width = diameter * spread_factor(vn)
    a_eff = max(alpha, MIN_ALPHA_DEG)
    length = width / np.sin(np.radians(a_eff))
    if np.linalg.norm(d3) > 0:
        gamma = float(np.degrees(np.arctan2(d3 @ rect.v, d3 @ rect.u)))
    else:
        gamma = float(rng.uniform(-180, 180))      # straight-on: no defined direction
        d3 = np.cos(np.radians(gamma)) * rect.u + np.sin(np.radians(gamma)) * rect.v
    tail = length * 0.45 * np.cos(np.radians(alpha)) * rng.uniform(0.6, 1.0) if kind == "parent" else 0.0
    uv = rect.to_local(hit_point)
    return Stain(sid, rect.id, event, droplet_id, kind, uv.tolist(), list(map(float, hit_point)),
                 float(width), float(length), float(alpha), gamma, list(map(float, d3)), float(tail), parent_id)


def satellites_for(stain, rect, speed, rng, scale=1.0):
    """Small secondary stains thrown off downstream of a parent stain.

    More and farther satellites for faster impacts; around a near-vertical drop
    they scatter in all directions (the crown of a drip stain).
    """
    lam = scale * 0.35 * max(speed - 2.0, 0.0)
    n = int(rng.poisson(lam))
    out = []
    g = np.radians(stain.gamma_deg)
    fwd = np.array([np.cos(g), np.sin(g)])
    side = np.array([-fwd[1], fwd[0]])
    round_ = stain.impact_angle_deg > 70
    for j in range(n):
        if round_:
            ang = rng.uniform(0, 2 * np.pi)
            dirv = np.array([np.cos(ang), np.sin(ang)])
            dist = stain.length / 2 + rng.uniform(1.0, 4.0) * stain.width
            off = dirv * dist
            gam = np.degrees(ang)
            alpha_s = rng.uniform(25, 60)
        else:
            dist = stain.length / 2 + stain.tail_length + rng.uniform(0.5, 3.0) * stain.width
            off = fwd * dist + side * rng.normal(0, 0.4) * stain.width
            gam = stain.gamma_deg + rng.normal(0, 4)
            alpha_s = np.clip(stain.impact_angle_deg + rng.normal(0, 4), 5, 85)
        w = stain.width * rng.uniform(0.12, 0.3)
        L = w / np.sin(np.radians(alpha_s))
        uv = np.asarray(stain.center_uv) + off
        if not rect.contains_local(uv):
            continue
        gr = np.radians(gam)
        d3 = np.cos(gr) * rect.u + np.sin(gr) * rect.v
        out.append(Stain(f"{stain.id}_s{j}", rect.id, stain.event, stain.droplet_id, "satellite",
                         uv.tolist(), rect.to_world(uv).tolist(), float(w), float(L), float(alpha_s),
                         float(gam), d3.tolist(), 0.0, stain.id))
    return out
