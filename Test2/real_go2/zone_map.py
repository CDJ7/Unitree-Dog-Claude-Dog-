#!/usr/bin/env python3
"""Zone map of the lab from the recording sessions, map-based localisation and route planning for goto_zone.py.

    python real_go2/zone_map.py           # build results/real_q1/zone_map.npz + zone_map.png from data_real/run_*
    python real_go2/zone_map.py --tune    # pick the localiser settings on the validation session
    python real_go2/zone_map.py --eval    # replay the held-out test session through the localiser
                                          #   -> results/real_q1/localization_eval.json + localization_eval.png

Why: the classifier alone confuses open_area with lab_2 / shelves (test accuracy 0.814), and one look-around
from a single spot is not enough to tell the zones apart. The recordings also hold the dog's odometry, so the
zones have a layout: lab_1 on one side, open_area in the middle, lab_2 and shelves on the other side.

* MAP: every session's odometry starts at a different place and heading, so the sessions are aligned to one
  another (rotation + shift that best matches their zone labels), then the labelled positions are blurred into
  a grid: P(zone | position), how well each cell is known, and where the dog has walked (walkable).
* LOCALISATION (ZoneLocalizer): a histogram filter over "where in the map did this session's odometry start,
  and how is it rotated" (heading_bins rotations x every walkable start cell). Each classifier output is
  evidence: hypotheses that put the dog in a zone the camera agrees with gain weight. The odometry carries the
  evidence along as the dog walks, so after a few metres through different zones one hypothesis wins, even
  though single frames are wrong 19% of the time.
* ROUTE: once localised, a shortest walkable path (Dijkstra, keeping away from the edges of the walked area)
  to the core of the target zone. Arrival needs the map to agree, not only the camera.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt, gaussian_filter

TEST2 = Path(__file__).resolve().parents[1]

MAP_DEFAULTS = {
    "file": "results/real_q1/zone_map.npz",
    "res_m": 0.25,              # grid cell size
    "blur_m": 0.75,             # zone labels are spread this far (boundaries are fuzzy, odometry drifts)
    "walk_m": 0.5,              # cells within this distance of a recorded position are walkable
    "heading_bins": 72,         # localisation: 5 deg rotation steps
    "update_hz": 4.0,           # localisation updates per second at most (camera frames are highly correlated)
    "evidence_weight": 0.3,     # exponent per update: < 1 because consecutive frames are not independent
    "classifier_floor": 0.3,    # mix the classifier's probabilities with uniform: it is over-confident
    "off_map_like": 0.08,       # likelihood of being somewhere never walked (vs 0.25 for "no idea which zone")
    "diffuse_s": 6.0,           # blur the belief this often (odometry drift)
    "localized_conf": 0.5,      # belief mass within 0.75 m / 10 deg of the best guess to call it localised
    "lost_conf": 0.25,          # following a route: below this, stop and look around
    "arrive_prob": 0.6,         # arrival (only once localised): P(in the target zone) from the belief >= this ...
    "arrive_min_share": 0.2,    # ... and the camera agrees in at least this share of the look-around views
    "arrive_prob_unlocalized": 0.9,  # or, not localised yet: the belief is this sure of the zone and the camera
                                     # holds navigation.arrive_share of the views
    "core_prob": 0.8,           # route goal: walkable cells where P(target zone) >= this
    "lookahead_m": 1.2,         # route following: steer at the path point this far ahead
    "replan_s": 3.0,            # re-plan the route this often (the pose estimate keeps improving)
    "route_max_s": 90.0,        # one route leg at most this long, then look around
    "turn_in_place_deg": 50.0,  # route point further off than this: stop and turn first
    "still_weight": 1.0,        # evidence factor while the dog stands still (< still_m since the last update); 1.0 = off:
    "still_m": 0.15,            #   repeated views of one spot are not independent evidence
    "explore_leg_s": 10.0,      # not localised yet: exploring legs this long (more ground = more evidence)
}


def map_cfg(nav):
    out = dict(MAP_DEFAULTS)
    out.update((nav or {}).get("map") or {})
    return out


def rot(th):
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


def apply(xy, th, t):
    return np.asarray(xy, float) @ rot(th).T + np.asarray(t, float)


# ---------------------------------------------------------------------------------------------- data
def load_runs(data_dir, classes, names=None):
    """{run name: (xy (N,2), yaw (N,), zone index (N,), t (N,))} from data_real/run_*/states.csv."""
    out = {}
    for d in sorted(Path(data_dir).glob("run_*")):
        if names is not None and d.name not in names:
            continue
        f = d / "states.csv"
        if not f.exists():
            continue
        rows = [r for r in csv.DictReader(f.open()) if r["zone"] in classes]
        if len(rows) < 20:
            continue
        xy = np.array([[float(r["base_x"]), float(r["base_y"])] for r in rows])
        out[d.name] = (xy, np.array([float(r["base_yaw"]) for r in rows]),
                       np.array([classes.index(r["zone"]) for r in rows]), np.array([float(r["t"]) for r in rows]))
    return out


# ---------------------------------------------------------------------------------------------- alignment
def _label_grid(xy, lab, nz, res, blur_m, margin=3.0):
    lo = xy.min(0) - margin
    shape = tuple((np.ceil((xy.max(0) + margin - lo) / res)).astype(int) + 1)
    cnt = np.zeros(shape + (nz,))
    ij = np.floor((xy - lo) / res).astype(int)
    np.add.at(cnt, (ij[:, 0], ij[:, 1], lab), 1.0)
    cnt = gaussian_filter(cnt, (blur_m / res, blur_m / res, 0))
    tot = cnt.sum(2)
    frac = np.where(tot[..., None] > 0.02, cnt / np.maximum(tot, 1e-9)[..., None], 0.0)
    return lo, frac


def _agreement(r, lab, lo, frac, res, T):
    """Mean label agreement of points r (N,2) shifted by each translation in T (M,2)."""
    ij = np.floor((r[None] + T[:, None] - lo) / res).astype(int)
    ok = (ij[..., 0] >= 0) & (ij[..., 1] >= 0) & (ij[..., 0] < frac.shape[0]) & (ij[..., 1] < frac.shape[1])
    i, j = np.where(ok, ij[..., 0], 0), np.where(ok, ij[..., 1], 0)
    v = np.where(ok, frac[i, j, np.broadcast_to(lab, i.shape)], 0.0)
    return v.mean(1)


def align(xy, lab, ref_xy, ref_lab, nz, res=0.25, blur_m=0.75, max_points=400):
    """Rotation th and shift t that best put a session's labelled path (xy, lab) onto a reference path:
    apply(xy, th, t) ~ reference frame. Coarse search (4 deg, 0.5 m within +-5 m), then two refinements.
    Returns (agreement 0..1, th, t)."""
    lo, frac = _label_grid(ref_xy, ref_lab, nz, res, blur_m)
    step = max(1, len(xy) // max_points)
    p, lb = xy[::step], lab[::step]
    centre = ref_xy.mean(0)
    best = (-1.0, 0.0, np.zeros(2))
    offs = np.stack(np.meshgrid(np.arange(-5, 5.01, 0.5), np.arange(-5, 5.01, 0.5)), -1).reshape(-1, 2)
    for deg in range(0, 360, 4):
        th = math.radians(deg)
        r = p @ rot(th).T
        T = centre - r.mean(0) + offs
        sc = _agreement(r, lb, lo, frac, res, T)
        i = int(sc.argmax())
        if sc[i] > best[0]:
            best = (float(sc[i]), th, T[i])
    for dth_deg, span, st in ((1.0, 0.5, 0.125), (0.5, 0.25, 0.0625)):
        sc0, th0, t0 = best
        offs = np.stack(np.meshgrid(np.arange(-span, span + 1e-9, st), np.arange(-span, span + 1e-9, st)), -1).reshape(-1, 2)
        for d in np.arange(-4, 5) * dth_deg:
            th = th0 + math.radians(d)
            r = p @ rot(th).T
            T = t0 + offs
            sc = _agreement(r, lb, lo, frac, res, T)
            i = int(sc.argmax())
            if sc[i] > best[0]:
                best = (float(sc[i]), th, T[i])
    return best


# ---------------------------------------------------------------------------------------------- map
class ZoneMap:
    """Grid map in a common frame: prob (nx, ny, nz) = P(zone | cell), known (nx, ny) 0..1 = how much was
    recorded near the cell, walk (nx, ny) = the dog has walked within walk_m, clear (nx, ny) = distance (m) to
    the edge of the walkable area."""

    def __init__(self, classes, lo, res, prob, known, walk, clear, points=None, labels=None, info=None):
        self.classes, self.lo, self.res = list(classes), np.asarray(lo, float), float(res)
        self.prob, self.known, self.walk, self.clear = prob, known, walk.astype(bool), clear
        self.points, self.labels, self.info = points, labels, info or {}
        self.shape = self.walk.shape
        self._extra_cost = np.zeros(self.shape)

    # -------------------------------------------------------------- build / io
    @classmethod
    def from_points(cls, classes, xy, lab, res=0.25, blur_m=0.75, walk_m=0.5, margin=1.5, info=None):
        nz = len(classes)
        lo = xy.min(0) - margin
        shape = tuple((np.ceil((xy.max(0) + margin - lo) / res)).astype(int) + 1)
        ij = np.floor((xy - lo) / res).astype(int)
        cnt = np.zeros(shape + (nz,))
        np.add.at(cnt, (ij[:, 0], ij[:, 1], lab), 1.0)
        visited = cnt.sum(2) > 0
        cnt = gaussian_filter(cnt, (blur_m / res, blur_m / res, 0))
        tot = cnt.sum(2)
        prob = np.where(tot[..., None] > 1e-3, cnt / np.maximum(tot, 1e-9)[..., None], 1.0 / nz)
        known = np.clip(gaussian_filter(visited.astype(float), blur_m / res) * 6.0, 0, 1)
        walk = distance_transform_edt(~visited) * res <= walk_m
        clear = distance_transform_edt(walk) * res
        return cls(classes, lo, res, prob, known, walk, clear, xy, lab, info)

    @classmethod
    def build(cls, runs, classes, ref=None, res=0.25, blur_m=0.75, walk_m=0.5, log=print):
        """Align every session to a reference session (default: the one with the most frames), then refine each
        session against the map of all the others, and grid the result."""
        nz = len(classes)
        names = list(runs)
        ref = ref or max(names, key=lambda n: len(runs[n][0]))
        tf = {ref: (1.0, 0.0, np.zeros(2))}
        rxy, rlab = runs[ref][0], runs[ref][2]
        for n in names:
            if n != ref:
                tf[n] = align(runs[n][0], runs[n][2], rxy, rlab, nz, res, blur_m)
                log(f"  {n} -> {ref}: rotation {math.degrees(tf[n][1]):+.0f} deg, shift ({tf[n][2][0]:+.2f}, "
                    f"{tf[n][2][1]:+.2f}) m, zone agreement {tf[n][0]:.2f}")
        for n in names:                                    # refine against all the other sessions together
            if n == ref:
                continue
            oxy = np.concatenate([apply(runs[m][0], tf[m][1], tf[m][2]) for m in names if m != n])
            olab = np.concatenate([runs[m][2] for m in names if m != n])
            sc, th, t = align(apply(runs[n][0], tf[n][1], tf[n][2]), runs[n][2], oxy, olab, nz, res, blur_m)
            c_th, c_t = tf[n][1], tf[n][2]
            tf[n] = (sc, c_th + th, rot(th) @ c_t + t)
        xy = np.concatenate([apply(runs[n][0], tf[n][1], tf[n][2]) for n in names])
        lab = np.concatenate([runs[n][2] for n in names])
        info = {"runs": names, "reference": ref,
                "transforms": {n: {"rotation_deg": round(math.degrees(tf[n][1]), 1),
                                   "shift_m": [round(float(v), 3) for v in tf[n][2]],
                                   "zone_agreement": round(tf[n][0], 3)} for n in names}}
        return cls.from_points(classes, xy, lab, res, blur_m, walk_m, info=info), tf

    def save(self, path):
        np.savez_compressed(path, classes=np.array(self.classes), lo=self.lo, res=self.res, prob=self.prob,
                            known=self.known, walk=self.walk, clear=self.clear, points=self.points,
                            labels=self.labels, info=json.dumps(self.info))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls([str(c) for c in z["classes"]], z["lo"], float(z["res"]), z["prob"], z["known"], z["walk"],
                   z["clear"], z["points"], z["labels"], json.loads(str(z["info"])))

    # -------------------------------------------------------------- queries
    def cell(self, xy):
        return tuple(np.floor((np.asarray(xy, float) - self.lo) / self.res).astype(int))

    def centre(self, ij):
        return self.lo + (np.asarray(ij, float) + 0.5) * self.res

    def inside(self, ij):
        return 0 <= ij[0] < self.shape[0] and 0 <= ij[1] < self.shape[1]

    def zone_probs(self, xy):
        ij = self.cell(xy)
        if not self.inside(ij):
            return np.full(len(self.classes), 1.0 / len(self.classes))
        return self.prob[ij]

    def zone_at(self, xy):
        return int(np.argmax(self.zone_probs(xy)))

    def block(self, xy, radius_m=0.5, cost=8.0):
        """A route leg was blocked here (LiDAR): make the planner avoid it for the rest of the mission."""
        c = self.centre(np.indices(self.shape).transpose(1, 2, 0))
        d = np.hypot(c[..., 0] - xy[0], c[..., 1] - xy[1])
        self._extra_cost[d <= radius_m] += cost

    def clear_blocks(self):
        self._extra_cost[:] = 0.0

    def goal_cells(self, k, core_prob=0.8):
        g = self.walk & (self.prob[..., k] >= core_prob) & (self.known >= 0.5)
        deep = g & (distance_transform_edt(self.prob[..., k] >= core_prob) * self.res >= 0.75)
        if deep.any():                                     # at least 0.75 m inside the zone's border
            g = deep
        if not g.any():                                    # fall back to the most target-like walkable cells
            p = np.where(self.walk, self.prob[..., k], -1)
            g = p >= max(p.max() - 0.1, 0.5)
        return g

    def plan(self, start_xy, k, core_prob=0.8):
        """Shortest walkable path (list of map xy, start first) from start_xy to the core of zone k, or None.
        Step cost grows near the edge of the walked area (keep to the middle of the corridors)."""
        s = self.cell(start_xy)
        if not self.inside(s) or not self.walk[s]:         # snap to the nearest walkable cell within 1.5 m
            w = np.argwhere(self.walk)
            if not len(w):
                return None
            d = np.hypot(*(self.centre(w) - np.asarray(start_xy, float)).T)
            if d.min() > 1.5:
                return None
            s = tuple(w[int(d.argmin())])
        goal = self.goal_cells(k, core_prob)
        if goal[s]:
            return [self.centre(s)]
        edge_pen = 1.0 + 2.0 * np.clip(1.0 - self.clear / 0.5, 0, 1) + self._extra_cost
        dist = np.full(self.shape, np.inf)
        prev = {}
        dist[s] = 0.0
        pq = [(0.0, s)]
        nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        end = None
        while pq:
            d, c = heapq.heappop(pq)
            if d > dist[c]:
                continue
            if goal[c]:
                end = c
                break
            for di, dj, w in nbrs:
                n = (c[0] + di, c[1] + dj)
                if not self.inside(n) or not self.walk[n]:
                    continue
                nd = d + w * self.res * edge_pen[n]
                if nd < dist[n]:
                    dist[n], prev[n] = nd, c
                    heapq.heappush(pq, (nd, n))
        if end is None:
            return None
        cells = [end]
        while cells[-1] != s:
            cells.append(prev[cells[-1]])
        return [self.centre(c) for c in reversed(cells)]


def path_length(path):
    return float(sum(np.hypot(*(np.asarray(b) - np.asarray(a))) for a, b in zip(path, path[1:])))


def lookahead_point(path, xy, dist):
    """The point `dist` metres along the path past the path point nearest to xy (the last point at the end).
    Returns (point, metres of path left)."""
    p = np.asarray(path, float)
    i = int(np.argmin(np.hypot(*(p - np.asarray(xy, float)).T)))
    left, j = 0.0, i
    while j + 1 < len(p) and left < dist:
        left += float(np.hypot(*(p[j + 1] - p[j])))
        j += 1
    remain = float(sum(np.hypot(*(p[m + 1] - p[m])) for m in range(i, len(p) - 1)))
    return p[j], remain


# ---------------------------------------------------------------------------------------------- localisation
class ZoneLocalizer:
    """Histogram filter over (rotation th_k, map cell m0 where this session's odometry origin p0 lies).
    Map position of odometry position p: m0 + R(th_k) (p - p0); map heading = odometry yaw + th_k.
    Prior: the dog starts on a walkable cell, any heading."""

    def __init__(self, zmap: ZoneMap, cfg=None):
        self.m, self.cfg = zmap, dict(MAP_DEFAULTS, **(cfg or {}))
        self.nth = int(self.cfg["heading_bins"])
        self.ths = np.arange(self.nth) * 2 * math.pi / self.nth
        self.pad = max(self.m.shape)
        nz = len(self.m.classes)
        self._zpad = [np.pad(self.m.prob[..., z] * self.m.known + (1 - self.m.known) / nz, self.pad,
                             constant_values=1.0 / nz) for z in range(nz)]
        self.reset()

    def reset(self):
        prior = np.where(self.m.walk, 0.0, -np.inf)
        self.logp = np.broadcast_to(prior, (self.nth,) + self.m.shape).copy()
        self.logp -= np.log(np.exp(self.logp).sum())
        self.p0, self.pos, self.yaw, self._last_pos = None, None, 0.0, None
        self._last_t, self._last_diffuse, self.updates = None, None, 0
        self._est = self._zb = None

    # each hypothesis k: the current position lies (si, sj) cells away from its start cell m0
    def _shifts(self, pos):
        d = np.asarray(pos, float) - self.p0
        c, s = np.cos(self.ths), np.sin(self.ths)
        dx, dy = c * d[0] - s * d[1], s * d[0] + c * d[1]
        si = np.clip(np.round(dx / self.m.res).astype(int), -self.pad, self.pad)
        sj = np.clip(np.round(dy / self.m.res).astype(int), -self.pad, self.pad)
        return si, sj

    def _shifted(self, grid_pad, si, sj):
        """grid values at (m0 + shift) for every start cell m0: a window into the padded grid."""
        P, (nx, ny) = self.pad, self.m.shape
        return grid_pad[P + si:P + si + nx, P + sj:P + sj + ny]

    def observe_likelihood(self, prob):
        cfg = self.cfg
        q = (1 - cfg["classifier_floor"]) * np.asarray(prob, float) + cfg["classifier_floor"] / len(prob)
        inside = self.m.prob @ q
        return self.m.known * inside + (1 - self.m.known) * cfg["off_map_like"]

    def update(self, pos, yaw, prob, now):
        """Odometry position (x, y), yaw and one classifier output. Rate-limited to update_hz."""
        self.pos, self.yaw = np.asarray(pos, float)[:2], float(yaw)
        self._est = self._zb = None
        if self.p0 is None:
            self.p0 = self.pos.copy()
        if self._last_t is not None and now - self._last_t < 1.0 / self.cfg["update_hz"]:
            return False
        self._last_t = now
        if self._last_diffuse is None:
            self._last_diffuse = now
        elif now - self._last_diffuse >= self.cfg["diffuse_s"]:
            self._diffuse()
            self._last_diffuse = now
        L = np.log(np.maximum(self.observe_likelihood(prob), 1e-6))
        Lp = np.pad(L, self.pad, constant_values=math.log(self.cfg["off_map_like"]))
        si, sj = self._shifts(self.pos)
        a = self.cfg["evidence_weight"]
        if self._last_pos is not None and np.hypot(*(self.pos - self._last_pos)) < self.cfg["still_m"]:
            a *= self.cfg["still_weight"]
        else:
            self._last_pos = self.pos.copy()
        for k in range(self.nth):
            self.logp[k] += a * self._shifted(Lp, si[k], sj[k])
        self.logp -= self.logp[np.isfinite(self.logp)].max()
        self.logp -= np.log(np.exp(self.logp).sum())
        self.updates += 1
        return True

    def move(self, pos, yaw):
        """Odometry only (no new classifier output): keeps the current pose for estimate()."""
        self.pos, self.yaw, self._est, self._zb = np.asarray(pos, float)[:2], float(yaw), None, None
        if self.p0 is None:
            self.p0 = self.pos.copy()

    def _diffuse(self):
        p = np.exp(self.logp)
        p = gaussian_filter(p, (0.4, 0.5, 0.5), mode=("wrap", "constant", "constant"))
        p = p * self.m.walk + 1e-7 * self.m.walk / max(self.m.walk.sum(), 1) / self.nth
        p /= p.sum()
        with np.errstate(divide="ignore"):
            self.logp = np.log(p)

    def estimate(self):
        """dict(xy=map position now, heading=map heading now, rot=odometry->map rotation, conf=belief mass near
        the best guess) or None before the first update."""
        if self.p0 is None or self.pos is None:
            return None
        if self._est is not None:
            return self._est
        k, i, j = np.unravel_index(int(np.argmax(self.logp)), self.logp.shape)
        th = float(self.ths[k])
        m0 = self.m.centre((i, j))
        xy = m0 + rot(th) @ (self.pos - self.p0)
        p = np.exp(self.logp)
        ks = [(k + d) % self.nth for d in (-2, -1, 0, 1, 2)]
        conf = float(p[ks, max(i - 3, 0):i + 4, max(j - 3, 0):j + 4].sum())
        self._est = {"xy": xy, "heading": wrap(self.yaw + th), "rot": th, "conf": conf}
        return self._est

    @property
    def localized(self):
        e = self.estimate()
        return e is not None and e["conf"] >= self.cfg["localized_conf"]

    def zone_belief(self):
        """P(the dog is in each zone now), summed over every hypothesis."""
        nz = len(self.m.classes)
        if self.p0 is None:
            return np.full(nz, 1.0 / nz)
        if self._zb is None:
            p = np.exp(self.logp)
            si, sj = self._shifts(self.pos)
            out = np.array([sum(float((p[k] * self._shifted(g, si[k], sj[k])).sum()) for k in range(self.nth))
                            for g in self._zpad])
            self._zb = out / max(out.sum(), 1e-12)
        return self._zb

    def to_odom_heading(self, map_heading):
        e = self.estimate()
        return wrap(map_heading - e["rot"])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------------------------- CLI
def plot_map(zmap, path, extra=None, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a7de1", "#e8602c", "#1bb27f", "#f0a30a", "#9b59b6", "#555555"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ext = [zmap.lo[0], zmap.lo[0] + zmap.shape[0] * zmap.res, zmap.lo[1], zmap.lo[1] + zmap.shape[1] * zmap.res]
    rgb = np.ones(zmap.shape + (3,))
    k = zmap.prob.argmax(2)
    for z in range(len(zmap.classes)):
        c = np.array(matplotlib.colors.to_rgb(cols[z]))
        m = zmap.walk & (k == z)
        rgb[m] = 0.55 * c + 0.45
    ax.imshow(rgb.transpose(1, 0, 2), origin="lower", extent=ext)
    if zmap.points is not None:
        for z, name in enumerate(zmap.classes):
            m = zmap.labels == z
            ax.scatter(zmap.points[m, 0], zmap.points[m, 1], s=1, c=cols[z], label=name)
    if extra:
        extra(ax, cols)
    ax.set_aspect("equal")
    ax.set_xlabel("map x (m)"), ax.set_ylabel("map y (m)")
    ax.legend(loc="best", fontsize=8, markerscale=6)
    ax.set_title(title or "Zone map: all sessions aligned; shaded = walkable, colour = most likely zone", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _holdout(cfg, split):
    """(map sessions, held-out session, predictions csv) for split 'test' (map from every other session) or
    'val' (map from the training sessions only: neither val nor test)."""
    runs = load_runs(TEST2 / cfg["data_dir"], cfg["classes"])
    order = sorted(runs)
    held = order[-1] if split == "test" else order[-2]
    used = [n for n in order if n != held and (split == "test" or n != order[-1])]
    return runs, used, held, TEST2 / cfg["results_dir"] / f"resnet18/{split}_predictions.csv"


def replay(zmap, runs, held, pred_csv, classes, mcfg, gt=None):
    """Feed a held-out session's odometry and real classifier outputs through the localiser, frame by frame.
    gt: (rotation, shift) of the session into the map from label alignment (the reference path)."""
    if gt is None:
        sc, th, t = align(runs[held][0], runs[held][2], zmap.points, zmap.labels, len(classes), mcfg["res_m"],
                          mcfg["blur_m"])
        gt = (th, t, sc)
    th, t, _ = gt
    rows = [r for r in csv.DictReader(Path(pred_csv).open()) if r["run"] == held]
    loc = ZoneLocalizer(zmap, mcfg)
    rec = []
    for r in rows:
        tt = float(r["t"])
        pos = np.array([float(r["base_x"]), float(r["base_y"])])
        loc.update(pos, float(r["base_yaw"]), np.array([float(r["p_" + c]) for c in classes]), tt)
        e = loc.estimate()
        zb = loc.zone_belief()
        g = apply(pos[None], th, t)[0]
        rec.append({"t": tt, "true": int(r["true"]), "pred": int(r["pred"]), "belief": int(zb.argmax()),
                    "conf": e["conf"], "err_m": float(np.hypot(*(e["xy"] - g))),
                    "head_err_deg": abs(math.degrees(wrap(e["rot"] - th))), "est": e["xy"].tolist(), "gt": g.tolist()})
    return rec, gt


def summarize(rec, mcfg):
    def acc(rs, key):
        return round(float(np.mean([x[key] == x["true"] for x in rs])), 3) if rs else None
    t0 = rec[0]["t"]
    loc_i = next((i for i, x in enumerate(rec) if x["conf"] >= mcfg["localized_conf"]), None)
    after = rec[loc_i:] if loc_i is not None else []
    return {
        "frames": len(rec),
        "first_localized_s": round(rec[loc_i]["t"] - t0, 1) if loc_i is not None else None,
        "share_frames_localized": round(float(np.mean([x["conf"] >= mcfg["localized_conf"] for x in rec])), 3),
        "after_first_localized": {
            "frames": len(after),
            "median_position_error_m": round(float(np.median([x["err_m"] for x in after])), 2) if after else None,
            "share_position_error_below_1m": round(float(np.mean([x["err_m"] < 1.0 for x in after])), 3) if after else None,
            "median_rotation_error_deg": round(float(np.median([x["head_err_deg"] for x in after])), 1) if after else None,
            "zone_accuracy_classifier": acc(after, "pred"),
            "zone_accuracy_localizer": acc(after, "belief"),
        },
        "whole_session": {"zone_accuracy_classifier": acc(rec, "pred"), "zone_accuracy_localizer": acc(rec, "belief")},
    }


TUNE_GRID = {"evidence_weight": [0.1, 0.2, 0.3], "classifier_floor": [0.3, 0.5], "diffuse_s": [2.0, 6.0]}


def tune(cfg, mcfg, out):
    """Pick the localiser settings on the validation session (map from the training sessions only); the test
    session is not looked at. Score: zone accuracy of the belief + share of the time localised within 1 m."""
    import itertools
    classes = cfg["classes"]
    runs, used, held, pred = _holdout(cfg, "val")
    if not pred.exists():
        sys.exit(f"{pred.relative_to(TEST2)} missing: run  python real_go2/evaluate_real.py --split val")
    print(f"tuning on {held} (map from {', '.join(used)})")
    zmap, _ = ZoneMap.build({n: runs[n] for n in used}, classes, res=mcfg["res_m"], blur_m=mcfg["blur_m"],
                            walk_m=mcfg["walk_m"], log=lambda *_: None)
    gt, rows = None, []
    keys = list(TUNE_GRID)
    for vals in itertools.product(*TUNE_GRID.values()):
        c = dict(mcfg, **dict(zip(keys, vals)))
        rec, gt = replay(zmap, runs, held, pred, classes, c, gt)
        s = summarize(rec, c)
        good = float(np.mean([x["conf"] >= c["localized_conf"] and x["err_m"] < 1.0 for x in rec]))
        score = s["whole_session"]["zone_accuracy_localizer"] + good
        rows.append({"settings": dict(zip(keys, vals)), "score": round(score, 3), "localised_within_1m": round(good, 3),
                     **s})
        print(f"  {dict(zip(keys, vals))}: belief acc {s['whole_session']['zone_accuracy_localizer']}, "
              f"localised within 1 m {good:.2f}, first localised {s['first_localized_s']} s")
    best = max(rows, key=lambda r: r["score"])
    (out / "localization_tuning.json").write_text(json.dumps(
        {"tuned_on": held, "map_sessions": used, "grid": TUNE_GRID, "best": best, "all": rows}, indent=1))
    print(f"best on {held}: {best['settings']} -> put these under navigation.map in real_config.yaml\n"
          f"-> {(out / 'localization_tuning.json').relative_to(TEST2)}")


def evaluate(cfg, mcfg, out):
    """Hold-out check on real data: map from every session except the test one, then replay the test session's
    odometry and real classifier outputs (results/real_q1/resnet18/test_predictions.csv) through the localiser."""
    classes = cfg["classes"]
    runs, used, held, pred = _holdout(cfg, "test")
    print(f"map from {', '.join(used)}; test session {held}")
    zmap, _ = ZoneMap.build({n: runs[n] for n in used}, classes, res=mcfg["res_m"], blur_m=mcfg["blur_m"],
                            walk_m=mcfg["walk_m"])
    rec, (th, _, sc) = replay(zmap, runs, held, pred, classes, mcfg)
    res = {"test_session": held, "map_sessions": used, "map_config": mcfg,
           "reference_alignment": {"rotation_deg": round(math.degrees(th), 1), "zone_agreement": round(sc, 3)},
           **summarize(rec, mcfg)}
    (out / "localization_eval.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "map_config"}, indent=1))
    plot_eval(zmap, rec, res, classes, out / "localization_eval.png")
    print(f"-> {(out / 'localization_eval.json').relative_to(TEST2)}, localization_eval.png")


def plot_eval(zmap, rec, res, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2a7de1", "#e8602c", "#1bb27f", "#f0a30a"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.4]})
    walk = np.where(zmap.walk, 0.93, 1.0)
    ext = [zmap.lo[0], zmap.lo[0] + zmap.shape[0] * zmap.res, zmap.lo[1], zmap.lo[1] + zmap.shape[1] * zmap.res]
    ax[0].imshow(walk.T, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    for z, name in enumerate(classes):
        m = zmap.labels == z
        ax[0].scatter(zmap.points[m, 0], zmap.points[m, 1], s=1, c=cols[z], alpha=0.25)
    gt = np.array([x["gt"] for x in rec])
    est = np.array([x["est"] for x in rec])
    ok = np.array([x["conf"] >= res["map_config"]["localized_conf"] for x in rec])
    ax[0].plot(gt[:, 0], gt[:, 1], "k-", lw=1.5, label="test session (reference alignment)")
    ax[0].scatter(est[ok, 0], est[ok, 1], s=4, c="#d1495b", label="localiser estimate (when localised)")
    ax[0].set_aspect("equal"), ax[0].legend(fontsize=8, loc="best")
    ax[0].set_title(f"{res['test_session']}: localiser vs reference path", fontsize=10)
    t = np.array([x["t"] for x in rec]) - rec[0]["t"]
    tr = np.array([x["true"] for x in rec])
    ax[1].plot(t, tr, "k-", lw=2, label="true zone")
    pr = np.array([x["pred"] for x in rec])
    be = np.array([x["belief"] for x in rec])
    ax[1].scatter(t[pr != tr], pr[pr != tr] + 0.12, s=6, c="#d1495b", label="classifier wrong (frame)")
    ax[1].scatter(t[be != tr], be[be != tr] - 0.12, s=6, c="#7b2cbf", label="map belief wrong")
    if res["first_localized_s"] is not None:
        ax[1].axvline(res["first_localized_s"], color="#888", ls="--", lw=1, label="first localised")
    ax[1].set_yticks(range(len(classes)), classes)
    ax[1].set_xlabel("time in session (s)")
    a = res["after_first_localized"]
    ax[1].set_title(f"after localising: classifier {a['zone_accuracy_classifier']}, map belief "
                    f"{a['zone_accuracy_localizer']}", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(TEST2 / "real_go2/real_config.yaml"))
    ap.add_argument("--eval", action="store_true", help="hold-out localisation check on the test session")
    ap.add_argument("--tune", action="store_true", help="choose the localiser settings on the validation session")
    ap.add_argument("--ref", help="reference session for the alignment (default: the longest)")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    mcfg = map_cfg(cfg.get("navigation"))
    out = TEST2 / cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    if a.tune:
        tune(cfg, mcfg, out)
        return
    if a.eval:
        evaluate(cfg, mcfg, out)
        return
    runs = load_runs(TEST2 / cfg["data_dir"], cfg["classes"])
    if len(runs) < 1:
        sys.exit(f"no sessions with states.csv in {cfg['data_dir']}")
    print(f"aligning {len(runs)} sessions ...")
    zmap, _ = ZoneMap.build(runs, cfg["classes"], ref=a.ref, res=mcfg["res_m"], blur_m=mcfg["blur_m"],
                            walk_m=mcfg["walk_m"])
    for n, v in zmap.info["transforms"].items():
        print(f"  {n}: rotation {v['rotation_deg']:+.0f} deg, zone agreement {v['zone_agreement']:.2f}")
    path = TEST2 / mcfg["file"]
    zmap.save(path)
    plot_map(zmap, path.with_suffix(".png"))
    print(f"-> {path.relative_to(TEST2)} ({zmap.shape[0]} x {zmap.shape[1]} cells of {zmap.res} m), "
          f"{path.with_suffix('.png').relative_to(TEST2)}")


if __name__ == "__main__":
    main()
