"""Zone map: the robot tags its own LiDAR map with named zones from a camera zone classifier.

The zone classifier (Test 2 Q1: ResNet18, one camera frame -> P(zone)) was trained with labels that say
where the dog *stands*, but a single view shows where it *faces*. Test 2 found that this is the main
error on the real dog (open-area accuracy 0.24-0.90 depending on heading). So one frame is never trusted
on its own. Instead:

1. Every classified frame adds its probabilities to the coarse cell (default 0.25 m) under the robot,
   in one of `heading_bins` heading bins. A cell's zone belief is the mean over the heading bins it has
   been seen from, so standing in one place and facing one way cannot outvote looking around.
2. Beliefs are smoothed over neighbouring cells (3 x 3), which removes isolated mistakes.
3. Every free cell of the occupancy grid then takes the zone of the nearest tagged cell *through free
   space* (a flood fill that does not cross walls), so the whole reachable map is labelled.

`regions()` summarises each zone: area, centroid, mean confidence and a goal point (the free cell
deepest inside the zone and away from obstacles), which map-based go-to-zone drives to.
Pure numpy/scipy; works with navigation.occupancy.OccupancyGrid.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, uniform_filter

COLORS = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (164, 123, 232), (209, 73, 91),
          (90, 90, 90)]                                    # RGB per zone index


class ZoneMap:
    def __init__(self, grid, classes, cell=0.25, radius=0.30, heading_bins=8, min_frames=3):
        self.grid, self.classes = grid, list(classes)
        self.cell, self.radius, self.hb, self.min_frames = float(cell), float(radius), int(heading_bins), int(min_frames)
        h = grid.shape[0] * grid.res
        w = grid.shape[1] * grid.res
        self.shape = (int(np.ceil(h / cell)), int(np.ceil(w / cell)))
        self.psum = np.zeros((self.hb,) + self.shape + (len(self.classes),), np.float32)
        self.n = np.zeros((self.hb,) + self.shape, np.float32)
        self.frames = 0

    # ------------------------------------------------------------ evidence
    def _cell(self, xy):
        ij = np.floor((np.asarray(xy, float) - self.grid.origin) / self.cell).astype(int)
        return int(ij[1]), int(ij[0])

    def add(self, xy, yaw, prob, weight=1.0):
        """One classified frame taken at robot position xy (m) and heading yaw (rad)."""
        prob = np.asarray(prob, np.float32)
        b = int(np.floor(((yaw % (2 * np.pi)) / (2 * np.pi)) * self.hb)) % self.hb
        r0, c0 = self._cell(xy)
        k = int(np.ceil(self.radius / self.cell))
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                r, c = r0 + dr, c0 + dc
                if not (0 <= r < self.shape[0] and 0 <= c < self.shape[1]):
                    continue
                cx = self.grid.origin + (np.array([c, r]) + 0.5) * self.cell
                d = float(np.linalg.norm(cx - np.asarray(xy, float)))
                if d > self.radius + 0.5 * self.cell:
                    continue
                w = weight * (1.0 if (dr, dc) == (0, 0) else 0.5)
                self.psum[b, r, c] += w * prob
                self.n[b, r, c] += w
        self.frames += 1

    def cell_probs(self):
        """(probs (R, C, K), heading bins seen per cell (R, C), frames per cell (R, C))."""
        seen = self.n > 0
        mean = np.where(seen[..., None], self.psum / np.maximum(self.n, 1e-9)[..., None], 0.0)
        nb = seen.sum(0)
        probs = mean.sum(0) / np.maximum(nb, 1)[..., None]
        return probs, nb, self.n.sum(0)

    def coarse(self):
        """Smoothed coarse labels (R, C) (-1 = no evidence) and confidence (R, C)."""
        probs, nb, nf = self.cell_probs()
        ok = nf >= self.min_frames
        w = ok.astype(np.float32)
        sm = np.stack([uniform_filter(probs[..., k] * w, 3, mode="constant") for k in range(probs.shape[-1])], -1)
        den = uniform_filter(w, 3, mode="constant")
        sm = sm / np.maximum(den, 1e-9)[..., None]
        lab = np.where(ok, sm.argmax(-1), -1)
        conf = np.where(ok, sm.max(-1), 0.0)
        return lab, conf

    # ------------------------------------------------------------ labelling
    def labels(self):
        """Zone index per occupancy-grid cell (-1 = not free or not labelled) and confidence."""
        g = self.grid
        free = g.free()
        lab_c, conf_c = self.coarse()
        rr, cc = np.indices(g.shape)
        rc = np.minimum((rr * g.res / self.cell).astype(int), self.shape[0] - 1)
        ccc = np.minimum((cc * g.res / self.cell).astype(int), self.shape[1] - 1)
        lab = np.where(free, lab_c[rc, ccc], -1)
        conf = np.where(lab >= 0, conf_c[rc, ccc], 0.0)
        # flood fill the rest of the free space from the tagged cells, one cell per step, never through obstacles
        four = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        for _ in range(4 * max(g.shape)):
            todo = free & (lab < 0)
            if not todo.any():
                break
            grown = maximum_filter(lab, footprint=four, mode="constant", cval=-1)
            reach = todo & (grown >= 0)
            if not reach.any():
                break
            lab[reach] = grown[reach]
        return lab, conf

    def zone_at(self, xy):
        lab, conf = self.labels()
        r, c = self.grid.to_cell(xy)
        if not self.grid.inside(r, c):
            return -1, 0.0
        return int(lab[r, c]), float(conf[r, c])

    def regions(self, min_clear=0.45, labelled=None):
        """{zone: {area_m2, centroid, goal, confidence, tagged_m2}} for zones present in the map."""
        g = self.grid
        lab, conf = labelled if labelled is not None else self.labels()
        clear = g.obstacle_distance()
        out = {}
        for k, name in enumerate(self.classes):
            m = lab == k
            if not m.any():
                continue
            inner = distance_transform_edt(m) * g.res               # distance to the zone's edge
            score = np.minimum(inner, clear) * m
            ok = m & (clear >= min_clear)
            r, c = np.unravel_index(int(np.argmax(np.where(ok, score, -1) if ok.any() else score)), m.shape)
            rr, cc = np.nonzero(m)
            tagged = conf[m] > 0
            out[name] = {"index": k, "area_m2": round(float(m.sum()) * g.res ** 2, 2),
                         "tagged_m2": round(float(tagged.sum()) * g.res ** 2, 2),
                         "centroid": np.round(g.to_xy(rr.mean(), cc.mean()), 3).tolist(),
                         "goal": np.round(g.to_xy(r, c), 3).tolist(),
                         "confidence": round(float(conf[m][tagged].mean()) if tagged.any() else 0.0, 3)}
        return out

    # --------------------------------------------------------------- output
    def image(self, path_xy=None, robot=None, scale=4):
        """RGB uint8 picture (north up): unknown grey, obstacles black, free cells tinted by zone."""
        import cv2
        g = self.grid
        lab, conf = self.labels()
        img = np.full(g.shape + (3,), 200, np.uint8)
        img[g.free()] = (250, 250, 250)
        for k in range(len(self.classes)):
            m = lab == k
            col = np.array(COLORS[k % len(COLORS)], float)
            a = (0.35 + 0.45 * np.clip(conf[m], 0, 1))[:, None]
            img[m] = (a * col + (1 - a) * 250).astype(np.uint8)
        img[g.occupied()] = (20, 20, 20)
        img = np.ascontiguousarray(img[::-1])
        img = cv2.resize(img, (g.shape[1] * scale, g.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

        def px(xy):
            return (int((xy[0] - g.origin[0]) / g.res * scale), int((g.shape[0] - (xy[1] - g.origin[1]) / g.res) * scale))
        if path_xy is not None and len(path_xy) > 1:
            cv2.polylines(img, [np.array([px(p) for p in path_xy], np.int32)], False, (40, 40, 160), 1, cv2.LINE_AA)
        for name, r in self.regions(labelled=(lab, conf)).items():
            p = px(r["goal"])
            cv2.circle(img, p, 4, (0, 0, 0), -1)
            cv2.putText(img, name, (p[0] + 6, p[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        if robot is not None:
            p = px(robot[:2])
            cv2.circle(img, p, 5, (200, 30, 30), -1)
            cv2.line(img, p, (int(p[0] + 14 * np.cos(robot[2])), int(p[1] - 14 * np.sin(robot[2]))), (200, 30, 30), 2)
        return img

    def save(self, out_dir, path_xy=None, extra=None):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        g = self.grid
        np.savez_compressed(out / "map.npz", logodds=g.logodds, seen=g.seen, low_hits=g.low_hits, origin=g.origin,
                            res=g.res, z=np.array([g.z_min, g.z_max]), psum=self.psum, n=self.n,
                            path=np.asarray(path_xy if path_xy is not None else np.zeros((0, 2))))
        meta = {"classes": self.classes, "cell": self.cell, "radius": self.radius, "heading_bins": self.hb,
                "min_frames": self.min_frames, "frames": self.frames, "regions": self.regions()}
        meta.update(extra or {})
        (out / "zones.json").write_text(json.dumps(meta, indent=1))
        import cv2
        cv2.imwrite(str(out / "zone_map.png"), cv2.cvtColor(self.image(path_xy), cv2.COLOR_RGB2BGR))
        return out

    @classmethod
    def load(cls, out_dir):
        """(ZoneMap, path_xy) from save(); the occupancy grid comes back with it as .grid."""
        from navigation.occupancy import OccupancyGrid
        d = np.load(Path(out_dir) / "map.npz")
        meta = json.loads((Path(out_dir) / "zones.json").read_text())
        g = OccupancyGrid.__new__(OccupancyGrid)
        g.origin, g.res = d["origin"].astype(float), float(d["res"])
        g.logodds, g.seen, g.low_hits = d["logodds"], d["seen"], d["low_hits"]
        g.shape = g.logodds.shape
        g.z_min, g.z_max = (float(v) for v in d["z"])
        zm = cls(g, meta["classes"], meta["cell"], meta["radius"], meta["heading_bins"], meta["min_frames"])
        zm.psum, zm.n, zm.frames = d["psum"], d["n"], meta["frames"]
        return zm, d["path"]
