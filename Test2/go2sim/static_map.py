"""Ground-truth 2D occupancy map of a crime scene, made from the compiled MuJoCo scene.

For data collection (Q2) and the RL layout export (Q3) we know the scene exactly, so instead of
mapping with the LiDAR (Phase 2's explorer does that) we rasterise it on a 5 cm grid:
- every collision geom outside the robot (house collider boxes, scenario props, mannequins) is
  turned into its world-frame footprint (the 2D convex hull of its rotated bounding box, exact for
  the house's box colliders) and its height range;
- a geom blocks the dog when it spans the dog's body height band (bottom below 0.45 m, top above
  3 cm), so rugs and mats stay walkable and door lintels or wall shelves do not block;
- flat geoms with their top at floor level are the floor; cells without floor are outside.
The result is loaded into Phase 2's `navigation.occupancy.OccupancyGrid`, so its costmap (inflation
by the robot radius) and planner (`navigation.planner.CostField`, Dijkstra via scikit-image) are
reused unchanged. `obstacles` keeps every footprint for the Q3 layout export.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

from go2sim import paths  # noqa: F401

import mujoco  # noqa: E402

from navigation.occupancy import L_MAX, L_MIN, OccupancyGrid  # noqa: E402
from navigation.planner import CostField, nearest_reachable, simplify  # noqa: E402

ORIGIN = (-0.5, -0.5)
SIZE = (9.2, 5.5)            # covers house 185 (x 0-8.2, y 0-4.5) with a margin
RES = 0.05
BODY_TOP = 0.45              # geoms whose bottom is above this pass over the Go2
OBSTACLE_Z = 0.03            # geoms whose top is below this are floor / rugs


def geom_footprints(model, data, robot_geoms):
    """[(name, kind, hull_xy (K,2), zmin, zmax)] for all collision geoms outside the robot."""
    out = []
    collide = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    for g in range(model.ngeom):
        if not collide[g] or g in robot_geoms or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        R = data.geom_xmat[g].reshape(3, 3)
        corners = (c + signs * h) @ R.T + data.geom_xpos[g]
        zmin, zmax = corners[:, 2].min(), corners[:, 2].max()
        xy = corners[:, :2]
        try:
            hull = xy[ConvexHull(xy).vertices]
        except Exception:
            hull = xy
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or \
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or f"geom{g}"
        if zmax <= OBSTACLE_Z + 0.02 and "floor" in name or (zmax <= OBSTACLE_Z and zmin < -0.01):
            kind = "floor"
        elif zmin < BODY_TOP and zmax > OBSTACLE_Z:
            kind = "obstacle"
        else:
            kind = "overhead_or_flat"
        out.append((name, kind, hull, float(zmin), float(zmax)))
    return out


def _raster(grid, hull):
    rows, cols = grid.shape
    lo = np.floor((hull.min(0) - grid.origin) / grid.res).astype(int) - 1
    hi = np.ceil((hull.max(0) - grid.origin) / grid.res).astype(int) + 1
    c0, r0 = max(lo[0], 0), max(lo[1], 0)
    c1, r1 = min(hi[0], cols), min(hi[1], rows)
    if c1 <= c0 or r1 <= r0:
        return None
    rr, cc = np.mgrid[r0:r1, c0:c1]
    pts = grid.to_xy(rr, cc).reshape(-1, 2)
    inside = MplPath(hull).contains_points(pts, radius=1e-9).reshape(rr.shape)
    return rr[inside], cc[inside]


def build_grid(model, data, robot_geoms, origin=ORIGIN, size=SIZE, res=RES):
    """(OccupancyGrid, blocked bool[rows, cols], footprints) of the scene."""
    grid = OccupancyGrid(origin=origin, size=size, res=res)
    floor = np.zeros(grid.shape, bool)
    obst = np.zeros(grid.shape, bool)
    fps = geom_footprints(model, data, robot_geoms)
    for name, kind, hull, zmin, zmax in fps:
        if kind == "overhead_or_flat":
            continue
        rc = _raster(grid, hull)
        if rc is None:
            continue
        (floor if kind == "floor" else obst)[rc] = True
    blocked = obst | ~floor
    grid.logodds[:] = np.where(blocked, L_MAX, L_MIN)
    grid.seen[:] = True
    return grid, blocked, fps


class StaticMap:
    """Occupancy grid + costmap + planner for one compiled scene."""

    def __init__(self, model, data, robot_geoms, robot_xy=None, robot_radius=0.24, soft=0.75):
        self.grid, self.blocked, self.footprints = build_grid(model, data, robot_geoms)
        self.robot_radius = robot_radius
        self.cost = self.grid.costmap(robot_radius=robot_radius, soft=soft)
        self.free = np.isfinite(self.cost)
        self.clearance = self.grid.obstacle_distance()

    def is_free(self, xy, margin=0.0):
        r, c = self.grid.to_cell(xy)
        if not self.grid.inside(r, c):
            return False
        return bool(self.free[r, c]) and self.clearance[r, c] >= self.robot_radius + margin

    def plan(self, start_xy, goal_xy, spacing=0.15):
        """Waypoints (N, 2) from start to (the reachable cell nearest) goal, or None."""
        field = CostField(self.grid, self.cost, start_xy)
        rc = self.grid.to_cell(goal_xy)
        if not (self.grid.inside(*rc) and np.isfinite(field.cum[rc])):
            rc = nearest_reachable(field, goal_xy, max_radius=0.8)
            if rc is None:
                return None
        return simplify(field.path_to(rc), spacing)

    def clearance_at(self, xy):
        r, c = self.grid.to_cell(xy)
        return float(self.clearance[r, c]) if self.grid.inside(r, c) else 0.0

    def sample_free(self, rng, polygon=None, min_clear=0.45, reachable=None, n_try=5000):
        """A random point with at least `min_clear` m to any obstacle, optionally inside a polygon
        and inside a reachability mask. Returns None if none found."""
        ok = self.clearance >= min_clear
        if reachable is not None:
            ok &= reachable
        rr, cc = np.nonzero(ok)
        if len(rr) == 0:
            return None
        pts = self.grid.to_xy(rr, cc)
        if polygon is not None:
            pts = pts[MplPath(np.asarray(polygon)).contains_points(pts)]
        if len(pts) == 0:
            return None
        return pts[rng.integers(len(pts))] + rng.uniform(-0.02, 0.02, 2)

    def reachable_from(self, start_xy):
        return CostField(self.grid, self.cost, start_xy).reachable()

    def extent(self):
        """(xmin, xmax, ymin, ymax) for imshow."""
        o, (h, w) = self.grid.origin, self.grid.shape
        return (o[0], o[0] + w * self.grid.res, o[1], o[1] + h * self.grid.res)
