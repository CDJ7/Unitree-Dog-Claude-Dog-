"""Camera coverage: which surfaces has the robot's camera actually looked at?

The LiDAR maps a room from one spot, but stains are only visible to the
camera, within a few metres and inside its field of view. This tracks, on
the occupancy grid, the cells the front camera has seen (horizontal FOV
76 deg, up to `max_range`, line of sight on the map), and plans a patrol:
viewpoints that see the most not-yet-seen surface (walls and furniture faces
next to free space, and floor), visited greedily, nearest-best first.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from navigation.local import turn_towards


class CameraCoverage:
    def __init__(self, grid, hfov_deg=75.9, max_range=3.0, floor_range=(0.7, 2.5)):
        self.grid = grid
        self.hfov = np.radians(hfov_deg)
        self.max_range = max_range
        self.floor_range = floor_range          # floor is only in view between these distances
        self.seen = np.zeros(grid.shape, bool)

    # ------------------------------------------------------------ visibility
    def visible(self, xy, yaw=None, fov=None, max_range=None, n_rays=None):
        """Cells visible from xy (LOS on the map): (rows, cols, is_floor_in_range)."""
        g = self.grid
        max_range = max_range or self.max_range
        fov = 2 * np.pi if yaw is None else (fov or self.hfov)
        n_rays = n_rays or max(16, int(np.degrees(fov) / 1.5))
        yaw = 0.0 if yaw is None else yaw
        ang = yaw + np.linspace(-fov / 2, fov / 2, n_rays)
        step = g.res * 0.5
        t = np.arange(1, int(max_range / step)) * step
        pts = np.asarray(xy)[None, None, :] + np.stack([np.cos(ang), np.sin(ang)], -1)[:, None, :] * t[None, :, None]
        r, c = g.to_cell(pts)
        ok = g.inside(r, c)
        r, c = np.where(ok, r, 0), np.where(ok, c, 0)
        occ = g.occupied()[r, c] & ok
        first = np.where(occ.any(1), occ.argmax(1), len(t))         # first occupied sample per ray
        keep = (np.arange(len(t))[None, :] <= first[:, None]) & ok
        rr, cc = r[keep], c[keep]
        dist = np.broadcast_to(t[None, :], keep.shape)[keep]
        floor_ok = (dist >= self.floor_range[0]) & (dist <= self.floor_range[1])
        return rr, cc, floor_ok

    def camera_pose_2d(self, sim):
        T = sim.camera_pose()
        z = T[:3, 2]
        return T[:2, 3], float(np.arctan2(z[1], z[0]))

    def update(self, sim):
        xy, yaw = self.camera_pose_2d(sim)
        rr, cc, floor_ok = self.visible(xy, yaw)
        occ = self.grid.occupied()[rr, cc]
        sel = occ | floor_ok
        self.seen[rr[sel], cc[sel]] = True

    # -------------------------------------------------------------- targets
    def targets(self):
        """Cells worth looking at: surfaces facing free space, and free floor."""
        free = self.grid.free()
        occ = self.grid.occupied()
        faces = occ & binary_dilation(free, np.ones((3, 3), bool))
        return faces, free

    def coverage(self):
        faces, floor = self.targets()
        f = (self.seen & faces).sum() / max(faces.sum(), 1)
        fl = (self.seen & floor).sum() / max(floor.sum(), 1)
        return {"surfaces": round(float(f), 3), "floor": round(float(fl), 3)}

    def gain(self, xy, unseen):
        rr, cc, floor_ok = self.visible(xy, None, max_range=2.5, n_rays=120)
        hit = unseen[rr, cc] & (self.grid.occupied()[rr, cc] | floor_ok)
        cells = np.unique(np.stack([rr[hit], cc[hit]], 1), axis=0) if hit.any() else np.zeros((0, 2), int)
        return cells

    def next_viewpoint(self, field, spacing=0.45, min_clear=0.50, min_gain=25, travel_weight=8.0, avoid=()):
        """(xy, look_at_xy, gain) of the best next viewpoint, or None when coverage is done.

        avoid: viewpoints already visited (candidates within 0.6 m are skipped)."""
        g = self.grid
        faces, floor = self.targets()
        unseen = (faces | floor) & ~self.seen
        clear = g.obstacle_distance()
        reach = np.isfinite(field.cum) & (clear >= min_clear)
        k = max(1, int(spacing / g.res))
        cand = np.argwhere(reach[::k, ::k]) * k
        best = None
        for r, c in cand:
            xy = g.to_xy(r, c)
            if any(np.linalg.norm(xy - np.asarray(v)) < 0.6 for v in avoid):
                continue
            cells = self.gain(xy, unseen)
            if len(cells) < min_gain:
                continue
            score = len(cells) / (1.0 + field.cum[r, c] * g.res / travel_weight)
            if best is None or score > best[2]:
                look = g.to_xy(*cells.mean(axis=0))
                best = (xy, look, score, len(cells))
        if best is None:
            return None
        return best[0], best[1], best[3]


def patrol(nav, cov, max_time=300.0, max_viewpoints=25, sweep_deg=40.0):
    """Visit viewpoints until the camera has seen (nearly) every reachable surface."""
    t0 = nav.sim.time
    visited = 0
    done_xy = []
    while nav.sim.time - t0 < max_time and visited < max_viewpoints:
        pick = cov.next_viewpoint(nav.field(), avoid=done_xy)
        if pick is None:
            nav.say(f"camera coverage complete: {cov.coverage()}")
            break
        xy, look, gain = pick
        nav.say(f"viewpoint {visited + 1}: {np.round(xy, 2).tolist()} ({gain} unseen cells), coverage {cov.coverage()}")
        ok = nav.go_to(xy, timeout=60.0, tol=0.3, final_tol=0.2)
        nav.log.goals.append((nav.sim.time, "viewpoint", float(xy[0]), float(xy[1]), bool(ok)))
        visited += 1
        done_xy.append(xy)
        planned = cov.gain(xy, ~cov.seen)
        if not ok:
            cov.seen[planned[:, 0], planned[:, 1]] = True      # do not try for these again
            continue
        nav.turn_to(look)
        st = nav.sim.state()
        for off in (sweep_deg, -sweep_deg, 0.0):
            yaw = st.yaw + np.radians(off)
            target = st.position[:2] + np.array([np.cos(yaw), np.sin(yaw)])
            nav.turn_to(target, timeout=6.0)
        cov.seen[planned[:, 0], planned[:, 1]] = True          # looked; leftovers are behind/occluded
    return cov.coverage()
