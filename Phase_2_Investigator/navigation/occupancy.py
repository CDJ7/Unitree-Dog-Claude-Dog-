"""2D occupancy grid built from the Go2's LiDAR (log-odds, 5 cm cells).

- The planar scan (720 rays at the LiDAR's height, ~0.45 m) carves free space
  along each ray and marks its end point occupied.
- The 3D cloud (MID-360, -7 to 52 deg) adds obstacles the scan plane misses:
  points between `z_min` and `z_max` above the floor (a lying mannequin, a
  low box, a table edge) go into a separate "low obstacle" layer that the
  scan cannot erase (the scan passes over them), since the scene is static.
  A cell needs 2 such hits. Floor points and the robot's own body are ignored.
  Note: the lowest MID-360 ray is 7 deg down, so something 0.2 m tall is only
  seen from more than ~2 m away; it stays in the map once seen.

Cells: unknown until observed; `free()` / `occupied()` threshold the log-odds.
The costmap for planning inflates occupied cells by the robot radius.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

L_FREE, L_OCC, L_MIN, L_MAX = -0.4, 0.85, -2.0, 3.5


class OccupancyGrid:
    def __init__(self, origin=(-1.5, -1.5), size=(13.0, 9.5), res=0.05, z_min=0.04, z_max=0.70):
        self.origin = np.asarray(origin, float)
        self.res = float(res)
        self.shape = (int(round(size[1] / res)), int(round(size[0] / res)))      # rows (y), cols (x)
        self.logodds = np.zeros(self.shape, np.float32)
        self.seen = np.zeros(self.shape, bool)
        self.low_hits = np.zeros(self.shape, np.int16)
        self.z_min, self.z_max = z_min, z_max

    # ------------------------------------------------------------- indexing
    def to_cell(self, xy):
        xy = np.asarray(xy, float)
        ij = np.floor((xy - self.origin) / self.res).astype(int)
        return ij[..., 1], ij[..., 0]                    # row, col

    def to_xy(self, rows, cols):
        return np.stack([self.origin[0] + (np.asarray(cols) + 0.5) * self.res,
                         self.origin[1] + (np.asarray(rows) + 0.5) * self.res], axis=-1)

    def inside(self, r, c):
        return (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])

    # --------------------------------------------------------------- update
    def _add(self, r, c, delta):
        ok = self.inside(r, c)
        r, c = r[ok], c[ok]
        np.add.at(self.logodds, (r, c), delta)
        self.seen[r, c] = True

    def update(self, scan, max_range=8.0):
        """Integrate one `interfaces.LidarScan` (world-frame poses inside)."""
        T = scan.T_world_lidar
        o = T[:3, 3]
        ang = scan.angles
        rng = scan.ranges.astype(float)
        hit = np.isfinite(rng) & (rng <= max_range)
        reach = np.where(hit, rng, max_range)
        d_local = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], -1)
        d_world = d_local @ T[:3, :3].T
        d_xy = d_world[:, :2] / np.maximum(np.linalg.norm(d_world[:, :2], axis=1, keepdims=True), 1e-9)
        # free space: samples along each ray up to 1 cell before the hit
        step = self.res * 0.5
        n = int(max_range / step)
        t = np.arange(n) * step
        mask = t[None, :] < (reach[:, None] - self.res)
        pts = o[None, None, :2] + d_xy[:, None, :] * t[None, :, None]
        pts = pts[mask]
        r, c = self.to_cell(pts)
        cells = np.unique(np.stack([r, c], 1), axis=0)
        self._add(cells[:, 0], cells[:, 1], L_FREE)
        # occupied: scan end points
        end = o[None, :2] + d_xy[hit] * rng[hit, None]
        r, c = self.to_cell(end)
        cells = np.unique(np.stack([r, c], 1), axis=0)
        self._add(cells[:, 0], cells[:, 1], L_OCC)
        # low obstacles from the 3D cloud
        pw = scan.points_world()
        dist = np.linalg.norm(pw[:, :2] - o[:2], axis=1)
        band = (pw[:, 2] > self.z_min) & (pw[:, 2] < self.z_max) & (dist < max_range) & (dist > 0.45)
        if band.any():
            r, c = self.to_cell(pw[band, :2])
            cells = np.unique(np.stack([r, c], 1), axis=0)
            ok = self.inside(cells[:, 0], cells[:, 1])
            cells = cells[ok]
            self.low_hits[cells[:, 0], cells[:, 1]] += 1
            self.seen[cells[:, 0], cells[:, 1]] = True
        np.clip(self.logodds, L_MIN, L_MAX, out=self.logodds)

    def add_low_points(self, pw, origin_xy, max_range=3.0):
        """Obstacle points from another sensor (the depth camera) into the low-obstacle layer."""
        dist = np.linalg.norm(pw[:, :2] - origin_xy, axis=1)
        sel = (pw[:, 2] > 0.03) & (pw[:, 2] < self.z_max) & (dist < max_range)
        if not sel.any():
            return
        r, c = self.to_cell(pw[sel, :2])
        cells = np.unique(np.stack([r, c], 1), axis=0)
        cells = cells[self.inside(cells[:, 0], cells[:, 1])]
        self.low_hits[cells[:, 0], cells[:, 1]] += 1
        self.seen[cells[:, 0], cells[:, 1]] = True

    def clear_disk(self, xy, radius):
        """Mark a disk free (the robot's own footprint at start)."""
        r0, c0 = self.to_cell(xy)
        k = int(radius / self.res) + 1
        rr, cc = np.mgrid[r0 - k:r0 + k + 1, c0 - k:c0 + k + 1]
        ok = self.inside(rr, cc) & ((rr - r0) ** 2 + (cc - c0) ** 2 <= k * k)
        self.logodds[rr[ok], cc[ok]] = np.minimum(self.logodds[rr[ok], cc[ok]], L_FREE * 3)
        self.seen[rr[ok], cc[ok]] = True

    # ---------------------------------------------------------------- views
    def low_obstacles(self):
        return self.low_hits >= 2

    def occupied(self):
        return (self.logodds > 0.6) | self.low_obstacles()

    def free(self):
        return self.seen & (self.logodds < -0.3) & ~self.low_obstacles()

    def unknown(self):
        return (~self.seen | ((self.logodds >= -0.3) & (self.logodds <= 0.6))) & ~self.low_obstacles()

    def occupied_points_near(self, xy, radius=1.5):
        """World xy of occupied cells within radius (for the safety layer)."""
        r0, c0 = self.to_cell(xy)
        k = int(radius / self.res)
        sl = (slice(max(r0 - k, 0), r0 + k + 1), slice(max(c0 - k, 0), c0 + k + 1))
        rr, cc = np.nonzero(self.occupied()[sl])
        return self.to_xy(rr + sl[0].start, cc + sl[1].start)

    def obstacle_distance(self):
        """Distance (m) from every cell to the nearest occupied cell."""
        occ = self.occupied()
        if not occ.any():
            return np.full(self.shape, 99.0)
        return distance_transform_edt(~occ) * self.res

    def costmap(self, robot_radius=0.28, soft=0.6, unknown_cost=None):
        """Traversal cost per cell for planning: inf inside the inflated obstacles,
        1 + a soft penalty near obstacles; unknown cells are inf unless unknown_cost is given."""
        dist = self.obstacle_distance()
        cost = 1.0 + 4.0 * np.clip((soft - dist) / (soft - robot_radius), 0, 1) ** 2
        cost[dist <= robot_radius] = np.inf
        unk = self.unknown()
        cost[unk] = np.inf if unknown_cost is None else np.maximum(cost[unk], unknown_cost)
        return cost
