"""Pure functions for the approach behaviour: target choice, steering, and the walking-corridor check."""
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

ALLOWED_ACTIONS = ("sit", "hello", "stretch", "content")


@dataclass
class Target:
    cls: str
    x: float            # forward (m), robot body frame
    y: float            # left (m)
    conf: float
    t: float            # time.time() of the result it came from

    @property
    def range(self):
        return math.hypot(self.x, self.y)

    @property
    def bearing(self):  # +left, radians
        return math.atan2(self.y, self.x)


def candidates(detections, classes, min_conf, t):
    """Detections of the wanted classes that have a position in the robot frame, as Targets."""
    out = []
    for d in detections:
        if d.name in classes and d.xyz_body is not None and d.conf >= min_conf:
            x, y = d.xyz_body[0], d.xyz_body[1]
            if np.isfinite(x) and np.isfinite(y):
                out.append(Target(d.name, float(x), float(y), float(d.conf), t))
    return out


def pick_target(cands: List[Target], prev: Optional[Target] = None, gate_bearing=0.6, gate_range=1.0):
    """No previous target: the nearest candidate. With one: the same-class candidate closest to where it was
    (bearing/range gating, so the dog does not jump to another chair when one disappears)."""
    if not cands:
        return None
    if prev is None:
        return min(cands, key=lambda c: c.range)
    near = [c for c in cands if c.cls == prev.cls and abs(c.bearing - prev.bearing) <= gate_bearing
            and abs(c.range - prev.range) <= gate_range]
    if not near:
        return None
    return min(near, key=lambda c: abs(c.bearing - prev.bearing) + abs(c.range - prev.range))


def approach_command(target: Target, stop_distance, p):
    """(vx, vyaw, arrived). Turn in place until roughly facing the target, then walk forward slowing as it nears.
    p: the `safety` config section. Never strafes, never reverses."""
    err = target.range - stop_distance
    bearing = target.bearing
    arrived = err <= p["stop_tolerance_m"]
    vyaw = 0.0 if abs(bearing) < 0.04 else max(-p["max_vyaw"], min(p["max_vyaw"], 1.0 * bearing))
    if arrived:
        return 0.0, vyaw if abs(bearing) > p["heading_tolerance_rad"] else 0.0, True
    if abs(bearing) > p["heading_tolerance_rad"]:
        return 0.0, vyaw, False
    vx = min(p["max_vx"], max(0.08, 0.45 * err))
    return vx, vyaw, False


def corridor_obstacle_distance(points_body, corridor, exclude_xy=None):
    """Distance ahead (m) of the nearest obstacle in the walking corridor, or None when it is clear.

    points_body: (N,3) in the robot body frame. Points near exclude_xy (the target we are walking to) are ignored.
    Needs `min_points` points so single noisy returns do not stop the dog.
    """
    p = np.asarray(points_body, dtype=float)
    if p.size == 0:
        return None
    c = corridor
    m = (p[:, 0] >= c["x_min"]) & (p[:, 0] <= c["x_max"]) & (np.abs(p[:, 1]) <= c["half_width_m"]) \
        & (p[:, 2] >= c["z_min"]) & (p[:, 2] <= c["z_max"]) & np.isfinite(p).all(axis=1)
    if exclude_xy is not None:
        d = np.hypot(p[:, 0] - exclude_xy[0], p[:, 1] - exclude_xy[1])
        m &= d > c["exclude_target_radius_m"]
    hits = p[m]
    if len(hits) < c["min_points"]:
        return None
    return float(np.sort(hits[:, 0])[max(c["min_points"] - 1, 0)])   # k-th nearest, robust to a stray point


def validate_config(cfg):
    """Raise ValueError for anything unsafe; return {class: (action, stop_distance)} with floors applied."""
    s = cfg["safety"]
    if s["max_vx"] > 0.4 or s["max_vyaw"] > 0.8:
        raise ValueError("autonomous speeds must not exceed the teleop defaults (0.4 m/s, 0.8 rad/s)")
    out = {}
    for cls, v in cfg["classes"].items():
        if v["action"] not in ALLOWED_ACTIONS:
            raise ValueError(f"class '{cls}': action '{v['action']}' is not allowed (allowed: {', '.join(ALLOWED_ACTIONS)})")
        out[cls] = (v["action"], max(v["stop_distance_m"], s["min_stop_distance_m"]))
    return out
