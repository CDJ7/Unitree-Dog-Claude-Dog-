"""Short-term memory of nearby obstacles, so the dog does not forget an object it just stopped in front of.

The problem it fixes (lab, Nov 2026): the navigator stopped ~0.5 m from an object, turned on the spot to look
around, and the object then dropped out of the LiDAR's view: too close (blind zone / the dog's own body filter
in rt/utlidar/cloud_deskewed), too low, or behind the sensor while turning. `free_profile` reads "no points" as
"free", so after the turn the object was treated as gone and the dog could walk straight back into it.

The fix: obstacles are remembered in the ODOMETRY frame (they stay put while the dog turns), and a remembered
cell is only forgotten when the sensor positively sees through it, never just because it is out of view.

  * LiDAR hits (points inside the walking corridor's height band) add evidence to 2-D grid cells.
  * Carving: a remembered cell loses evidence only if, in this scan, it lies inside the LiDAR's reliable view
    (range between blind_m and carve_max_m, bearing inside fov_deg) AND the beam demonstrably passed through it:
    the nearest return in that direction, at or below the height the obstacle was seen at, is farther away than
    the cell. An EMPTY direction never carves (that is exactly what a too-close object looks like), and a beam
    passing over a low box to the wall behind it does not erase the box. Cells in the blind zone, behind the
    sensor, or in an unseen direction keep their evidence.
  * Time: unseen cells fade only after forget_s (default 120 s), long enough for any look-around.
  * Vision: camera detections (Test 1's YOLO + LiDAR fusion: Detection.xyz_body) become labelled OBJECTS
    ("chair 0.6 m behind-left"); they also add evidence around their position, so an object the LiDAR misses
    entirely (low, thin, dark) is still remembered once the camera saw it. A detection without a distance
    labels the nearest remembered LiDAR cluster on that bearing.

`free_profile(pose, ...)` gives the remembered free distance per body direction with the same corridor maths as
the live LiDAR (goto_zone.free_profile); the navigator uses min(live, remembered) for every decision.
Pure numpy, no robot needed: tested in tests/test_obstacle_memory.py and closed-loop in real_go2/nav_sim.py
(--blind: a LiDAR with a near blind zone and a forward-only field of view).
"""
from __future__ import annotations

import math

import numpy as np

MEMORY_DEFAULTS = {
    "enabled": True,
    "res_m": 0.05,            # grid cell size
    "z_min": -0.20,           # height band (body frame) that counts as an obstacle: the walking corridor's
    "z_max": 0.45,
    "max_range_m": 3.0,       # LiDAR hits farther than this are not stored (only near obstacles matter)
    "blind_m": 0.45,          # closer than this (from the dog's centre) the LiDAR is NOT trusted to see an object
    "fov_deg": 180.0,         # horizontal sector (centred straight ahead) where the LiDAR reliably sees obstacles
    "carve_max_m": 2.0,       # a cell can only be carved (seen through) within this range
    "hit": 1.0,               # evidence added per scan that hits a cell
    "miss": 0.35,             # evidence removed per scan that sees through a cell
    "max_score": 4.0,
    "occupied": 1.0,          # cells with at least this evidence are obstacles
    "forget_s": 120.0,        # unseen cells are dropped after this long (the dog has moved on)
    "bearing_bins": 72,       # 5 deg bins for carving
    "object_radius_m": 0.25,  # a camera detection marks a disc this size (if the LiDAR saw nothing there)
    "object_gate_m": 0.6,     # a new detection of the same label within this distance updates the same object
    "object_forget_s": 300.0,
    "bearing_label_m": 3.0,   # a detection without distance labels the nearest LiDAR cluster on its bearing up to this
}


# carving height slabs (top of each, body frame z): a cell whose lowest hit was at height z is only carved by
# returns from the first slab whose top is >= z + 5 cm
SLABS = np.array([-0.05, 0.10, 0.25, 0.45, 10.0])


def mem_cfg(nav):
    """Memory settings: MEMORY_DEFAULTS updated with navigation.memory from real_config.yaml."""
    return {**MEMORY_DEFAULTS, **((nav or {}).get("memory") or {})}


def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class ObstacleMemory:
    """Remembered obstacles around the dog, in the odometry frame. pose = (x, y, yaw) from odometry."""

    def __init__(self, cfg=None):
        self.cfg = {**MEMORY_DEFAULTS, **(cfg or {})}
        self.cells = {}        # (i, j) -> [evidence, last time hit or carved, lowest z seen (body frame)]
        self.objects = []      # dicts: id, label, xy (odom), conf, first, last, n
        self._next_id = 0

    # ------------------------------------------------------------------ helpers
    def _key(self, xy):
        r = self.cfg["res_m"]
        return int(math.floor(xy[0] / r)), int(math.floor(xy[1] / r))

    def _centre(self, k):
        r = self.cfg["res_m"]
        return np.array([(k[0] + 0.5) * r, (k[1] + 0.5) * r])

    @staticmethod
    def to_world(pts_xy, pose):
        return np.asarray(pts_xy, float).reshape(-1, 2) @ _rot(pose[2]).T + np.asarray(pose[:2], float)

    @staticmethod
    def to_body(pts_xy, pose):
        return (np.asarray(pts_xy, float).reshape(-1, 2) - np.asarray(pose[:2], float)) @ _rot(pose[2])

    def occupied_xy(self):
        """(N, 2) odom-frame centres of the cells that count as obstacles."""
        thr = self.cfg["occupied"]
        ks = [k for k, e in self.cells.items() if e[0] >= thr]
        return np.array([self._centre(k) for k in ks]).reshape(-1, 2)

    def clear(self):
        self.cells.clear()
        self.objects.clear()

    # ------------------------------------------------------------------ LiDAR
    def update_lidar(self, pts_body, pose, now):
        """One LiDAR scan: (N, 3) points in the body frame (x forward, y left, z up), pose = odometry (x, y, yaw)."""
        c = self.cfg
        p = np.asarray(pts_body, float).reshape(-1, 3)
        p = p[np.isfinite(p).all(1)]
        band = p[(p[:, 2] >= c["z_min"]) & (p[:, 2] <= c["z_max"])]
        rng = np.hypot(band[:, 0], band[:, 1])
        band, rng = band[rng <= c["max_range_m"]], rng[rng <= c["max_range_m"]]
        # 1) hits: evidence up, once per cell per scan
        hit_keys = set()
        if len(band):
            w = self.to_world(band[:, :2], pose)
            zmin = {}
            for q, z in zip(w, band[:, 2]):
                k = self._key(q)
                zmin[k] = min(zmin.get(k, np.inf), float(z))
            for k, z in zmin.items():
                hit_keys.add(k)
                e = self.cells.get(k)
                if e is None:
                    self.cells[k] = [c["hit"], now, z]
                else:
                    e[0], e[1], e[2] = min(e[0] + c["hit"], c["max_score"]), now, min(e[2], z)
        # 2) carving: per height slab and bearing bin, the nearest return at or below that height in this scan
        nbb = int(c["bearing_bins"])
        nearest = np.full((len(SLABS), nbb), np.inf)
        if len(band):
            b = (np.round(np.arctan2(band[:, 1], band[:, 0]) / (2 * math.pi) * nbb).astype(int)) % nbb
            for si, top in enumerate(SLABS):
                m = band[:, 2] <= top
                np.minimum.at(nearest[si], b[m], rng[m])
        self._carve(pose, now, nearest, hit_keys)
        # 3) forget what has not been seen for a long time
        self.cells = {k: e for k, e in self.cells.items() if now - e[1] <= c["forget_s"] and e[0] > 0}

    def _carve(self, pose, now, nearest, hit_keys):
        c = self.cfg
        if not self.cells:
            return
        keys = [k for k in self.cells if k not in hit_keys]
        if not keys:
            return
        cb = self.to_body(np.array([self._centre(k) for k in keys]), pose)
        r = np.hypot(cb[:, 0], cb[:, 1])
        ang = np.arctan2(cb[:, 1], cb[:, 0])
        nbb = nearest.shape[1]
        b = (np.round(ang / (2 * math.pi) * nbb).astype(int)) % nbb
        in_view = (r >= c["blind_m"]) & (r <= c["carve_max_m"]) & (np.abs(ang) <= math.radians(c["fov_deg"]) / 2)
        # seen through: a return FARTHER away in that direction, no higher than where the obstacle was seen
        # (+5 cm); an empty direction (inf) never counts
        z = np.array([self.cells[k][2] for k in keys])
        si = np.clip(np.searchsorted(SLABS, z + 0.05), 0, len(SLABS) - 1)
        margin = 2.5 * c["res_m"]
        near = nearest[si, b]
        through = np.isfinite(near) & (near > r + margin)
        for k, ok in zip(keys, in_view & through):
            if ok:
                e = self.cells[k]
                e[0] -= c["miss"]
                e[1] = now

    # ------------------------------------------------------------------ vision
    def add_detection(self, label, conf, pose, now, xyz_body=None, bearing=None):
        """A camera detection. With xyz_body (Test 1's LiDAR fusion) it becomes / updates a labelled object at that
        odom position and marks a disc of evidence there. With only a bearing (rad, body frame, left +), it labels the
        nearest remembered obstacle cluster in that direction (if any). Returns the object dict or None."""
        c = self.cfg
        if xyz_body is not None and np.all(np.isfinite(xyz_body[:2])):
            xy = self.to_world(np.asarray(xyz_body[:2], float), pose)[0]
        elif bearing is not None:
            occ = self.occupied_xy()
            if not len(occ):
                return None
            ob = self.to_body(occ, pose)
            r, a = np.hypot(ob[:, 0], ob[:, 1]), np.arctan2(ob[:, 1], ob[:, 0])
            ok = (np.abs(np.angle(np.exp(1j * (a - bearing)))) <= math.radians(8)) & (r <= c["bearing_label_m"])
            if not ok.any():
                return None
            xy = occ[np.flatnonzero(ok)[np.argmin(r[ok])]]
        else:
            return None
        obj = None
        for o in self.objects:
            if o["label"] == label and np.hypot(*(o["xy"] - xy)) <= c["object_gate_m"]:
                obj = o
                break
        if obj is None:
            obj = {"id": self._next_id, "label": label, "xy": np.array(xy, float), "conf": float(conf),
                   "first": now, "last": now, "n": 0}
            self._next_id += 1
            self.objects.append(obj)
        n = obj["n"]
        obj["xy"] = (obj["xy"] * n + xy) / (n + 1) if n else np.array(xy, float)
        obj["n"], obj["last"], obj["conf"] = min(n + 1, 20), now, max(obj["conf"] * 0.9, float(conf))
        if xyz_body is not None:                         # make sure the object blocks the way even if the LiDAR missed it
            rr, res = c["object_radius_m"], c["res_m"]
            for dx in np.arange(-rr, rr + 1e-9, res):
                for dy in np.arange(-rr, rr + 1e-9, res):
                    if dx * dx + dy * dy <= rr * rr:
                        k = self._key(obj["xy"] + (dx, dy))
                        e = self.cells.setdefault(k, [0.0, now, float(xyz_body[2]) if len(xyz_body) > 2
                                                      else c["z_min"]])
                        e[0], e[1] = max(e[0], c["occupied"]), now
        return obj

    def forget_objects(self, now):
        self.objects = [o for o in self.objects if now - o["last"] <= self.cfg["object_forget_s"]]

    # ------------------------------------------------------------------ queries
    def free_profile(self, pose, nb, corridor, lookahead):
        """Remembered free distance in nb body directions (same maths as the live LiDAR's free_profile)."""
        from real_go2.goto_zone import free_profile
        occ = self.occupied_xy()
        if not len(occ):
            return np.full(nb, float(lookahead))
        ob = self.to_body(occ, pose)
        pts = np.column_stack([ob, np.zeros(len(ob))])
        cor = dict(corridor, z_min=-1.0, z_max=1.0, min_points=1)
        return free_profile(pts, nb, cor, lookahead)

    def nearest(self, pose):
        """(distance from the dog's centre, bearing in the body frame, label or None) of the closest remembered
        obstacle, or None."""
        occ = self.occupied_xy()
        if not len(occ):
            return None
        ob = self.to_body(occ, pose)
        r = np.hypot(ob[:, 0], ob[:, 1])
        i = int(np.argmin(r))
        label = None
        for o in self.objects:
            if np.hypot(*(o["xy"] - occ[i])) <= self.cfg["object_gate_m"]:
                label = o["label"]
                break
        return float(r[i]), float(math.atan2(ob[i, 1], ob[i, 0])), label

    def objects_near(self, pose, within_m=3.0):
        """Labelled objects within within_m: list of (label, distance, body bearing, seconds since seen)."""
        out = []
        for o in self.objects:
            b = self.to_body(o["xy"], pose)[0]
            d = float(np.hypot(*b))
            if d <= within_m:
                out.append((o["label"], d, float(math.atan2(b[1], b[0])), o["last"]))
        return sorted(out, key=lambda t: t[1])


def describe_bearing(a):
    """Body bearing (rad, left +) -> 'ahead', 'ahead-left', 'left', 'behind-left', 'behind', ..."""
    d = math.degrees(a)
    names = ["ahead", "ahead-left", "left", "behind-left", "behind", "behind-right", "right", "ahead-right"]
    return names[int(round(d / 45.0)) % 8]
