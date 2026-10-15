"""Persistent object map: merges repeated sightings and scores the result against known positions."""
import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MapObject:
    id: int
    cls: str
    surface: deque = field(default_factory=lambda: deque(maxlen=400))     # observed surface points (world)
    centers: deque = field(default_factory=lambda: deque(maxlen=400))     # corrected towards the object centre
    ranges: deque = field(default_factory=lambda: deque(maxlen=400))
    viewpoints: list = field(default_factory=list)                         # coarse robot positions it was seen from
    n_obs: int = 0

    def position(self, corrected=True):
        pts = self.centers if corrected else self.surface
        return np.median(np.asarray(pts), axis=0)


class ObjectMap:
    def __init__(self, cfg):
        self.cfg = cfg
        self._objs: List[MapObject] = []
        self._next_id = 1

    def _half_depth(self, cls):
        return self.cfg["class_half_depth_m"].get(cls, self.cfg["default_half_depth_m"])

    def add(self, cls, xyz_world, range_m=None, ray_origin=None, robot_xy=None):
        """Add one sighting. ray_origin: where the sensor was (to push the surface point back to the centre)."""
        if range_m is not None and range_m > self.cfg["max_obs_range_m"]:
            return None
        p = np.asarray(xyz_world, dtype=float)
        if not np.isfinite(p).all():
            return None
        c = p.copy()
        if self.cfg["use_center_correction"] and ray_origin is not None:
            d = p - np.asarray(ray_origin, dtype=float)
            d[2] = 0.0                                   # push horizontally: objects stand on the floor
            n = np.linalg.norm(d)
            if n > 1e-6:
                c = p + d / n * self._half_depth(cls)
        best, best_d = None, self.cfg["merge_dist_m"]
        for o in self._objs:
            if o.cls != cls:
                continue
            dist = float(np.linalg.norm(o.position() - c))
            if dist < best_d:
                best, best_d = o, dist
        if best is None:
            best = MapObject(self._next_id, cls)
            self._next_id += 1
            self._objs.append(best)
        best.surface.append(p)
        best.centers.append(c)
        if range_m is not None:
            best.ranges.append(range_m)
        best.n_obs += 1
        if robot_xy is not None and all(math.hypot(robot_xy[0] - v[0], robot_xy[1] - v[1]) > 0.5 for v in best.viewpoints):
            best.viewpoints.append(tuple(robot_xy))
        return best

    def objects(self, confirmed_only=True) -> List[MapObject]:
        return [o for o in self._objs if not confirmed_only or o.n_obs >= self.cfg["min_observations"]]

    def estimates(self):
        """[(class, x, y)] of confirmed objects, for drawing."""
        return [(o.cls, *o.position()[:2]) for o in self.objects()]

    def to_dict(self):
        out = []
        for o in self.objects():
            out.append({"id": o.id, "class": o.cls,
                        "position_world_m": [round(float(v), 4) for v in o.position()],
                        "surface_position_world_m": [round(float(v), 4) for v in o.position(corrected=False)],
                        "n_observations": o.n_obs, "n_viewpoints": len(o.viewpoints),
                        "mean_range_m": round(float(np.mean(o.ranges)), 3) if o.ranges else None})
        return out


def evaluate(objmap: ObjectMap, truth, match_dist_m=1.0):
    """Compare confirmed map objects with true objects. truth: [(name, cls, center_xyz)].

    Greedy nearest same-class assignment, closest pairs first. Returns a dict with per-object errors
    (3D, XY, and the uncorrected surface-point error), misses and false positives.
    """
    est = objmap.objects()
    pairs = []
    for ti, (_, tcls, tc) in enumerate(truth):
        for ei, o in enumerate(est):
            if o.cls == tcls:
                d = float(np.linalg.norm(o.position() - np.asarray(tc)))
                if d <= match_dist_m:
                    pairs.append((d, ti, ei))
    pairs.sort()
    used_t, used_e, match = set(), set(), {}
    for d, ti, ei in pairs:
        if ti in used_t or ei in used_e:
            continue
        used_t.add(ti); used_e.add(ei); match[ti] = ei
    rows = []
    for ti, (name, cls, tc) in enumerate(truth):
        tc = np.asarray(tc, dtype=float)
        if ti in match:
            o = est[match[ti]]
            pe, ps = o.position(), o.position(corrected=False)
            rows.append({"object": name, "class": cls, "detected": True, "n_observations": o.n_obs,
                         "true_xyz": tc.round(3).tolist(), "est_xyz": pe.round(3).tolist(),
                         "error_3d_m": round(float(np.linalg.norm(pe - tc)), 3),
                         "error_xy_m": round(float(np.linalg.norm(pe[:2] - tc[:2])), 3),
                         "surface_error_3d_m": round(float(np.linalg.norm(ps - tc)), 3)})
        else:
            rows.append({"object": name, "class": cls, "detected": False, "true_xyz": tc.round(3).tolist()})
    fps = [{"class": o.cls, "est_xyz": o.position().round(3).tolist(), "n_observations": o.n_obs}
           for ei, o in enumerate(est) if ei not in used_e]
    errs = [r["error_3d_m"] for r in rows if r["detected"]]
    errs_xy = [r["error_xy_m"] for r in rows if r["detected"]]
    summary = {"n_true": len(truth), "n_detected": len(errs), "recall": round(len(errs) / max(len(truth), 1), 3),
               "n_false_positives": len(fps),
               "precision": round(len(errs) / max(len(est), 1), 3),
               "mean_error_3d_m": round(float(np.mean(errs)), 3) if errs else None,
               "median_error_3d_m": round(float(np.median(errs)), 3) if errs else None,
               "max_error_3d_m": round(float(np.max(errs)), 3) if errs else None,
               "mean_error_xy_m": round(float(np.mean(errs_xy)), 3) if errs_xy else None}
    return {"summary": summary, "per_object": rows, "false_positives": fps}
