"""Global path planning on the occupancy costmap (minimum-cost path, 8-connected).

Uses scikit-image's MCP_Geometric (Dijkstra in C): one call gives the cost-to-go
from the robot to every cell, which serves both path planning and frontier
selection. Infinite-cost cells (inflated obstacles, unknown space) are impassable.
"""
from __future__ import annotations

import numpy as np
from skimage.graph import MCP_Geometric


class CostField:
    """Cumulative traversal cost from the robot's cell to every cell."""

    def __init__(self, grid, cost, start_xy, escape_radius=0.35):
        self.grid = grid
        cost = cost.copy()
        r0, c0 = grid.to_cell(start_xy)
        # the robot may stand inside an inflated zone (near a wall): let it leave
        k = int(escape_radius / grid.res)
        rr, cc = np.mgrid[r0 - k:r0 + k + 1, c0 - k:c0 + k + 1]
        ok = grid.inside(rr, cc) & ((rr - r0) ** 2 + (cc - c0) ** 2 <= k * k)
        occ = grid.occupied()
        esc = ok.copy()
        esc[ok] &= ~np.isfinite(cost[rr[ok], cc[ok]]) & ~occ[rr[ok], cc[ok]]
        cost[rr[esc], cc[esc]] = 20.0
        self.cost = cost
        self.start = (int(r0), int(c0))
        self.mcp = MCP_Geometric(np.where(np.isfinite(cost), cost, -1.0), fully_connected=True)
        self.cum, _ = self.mcp.find_costs([self.start])

    def reachable(self):
        return np.isfinite(self.cum)

    def path_to(self, rc):
        """Waypoints (N, 2) in metres from the robot to cell rc, or None."""
        if not np.isfinite(self.cum[rc]):
            return None
        cells = np.array(self.mcp.traceback(rc))
        return self.grid.to_xy(cells[:, 0], cells[:, 1])

    def distance_to(self, rc):
        return float(self.cum[rc]) * self.grid.res


def simplify(path, spacing=0.15):
    """Resample a cell path to roughly even spacing (keeps the end point)."""
    if path is None or len(path) < 2:
        return path
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    s = np.arange(0, d[-1], spacing)
    out = np.column_stack([np.interp(s, d, path[:, 0]), np.interp(s, d, path[:, 1])])
    return np.vstack([out, path[-1]])


def nearest_reachable(field, xy, max_radius=1.0):
    """The reachable cell closest to xy (for goals inside obstacles' inflation)."""
    g = field.grid
    r0, c0 = g.to_cell(xy)
    k = int(max_radius / g.res)
    rr, cc = np.mgrid[r0 - k:r0 + k + 1, c0 - k:c0 + k + 1]
    ok = g.inside(rr, cc)
    rr, cc = rr[ok], cc[ok]
    reach = np.isfinite(field.cum[rr, cc])
    if not reach.any():
        return None
    rr, cc = rr[reach], cc[reach]
    i = int(np.argmin((rr - r0) ** 2 + (cc - c0) ** 2))
    return int(rr[i]), int(cc[i])
