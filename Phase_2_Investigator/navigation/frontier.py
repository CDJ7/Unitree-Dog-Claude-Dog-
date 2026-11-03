"""Frontier exploration: go to the boundary between known free space and unknown space.

Frontier cells are free cells with an unknown 4-neighbour. They are grouped into
8-connected clusters; small clusters (noise) are dropped. A cluster's score is
its path distance from the robot minus a bonus for its size (more unknown
space to uncover), so the robot explores nearby, large openings first.
Goals that failed are blacklisted. (Same idea as the simulator's
skills/explore frontier.py, rewritten here for our grid.)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label


def frontier_cells(grid):
    free, unk = grid.free(), grid.unknown()
    neigh_unknown = binary_dilation(unk, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    return free & neigh_unknown


def frontier_clusters(grid, min_cells=6):
    labels, n = label(frontier_cells(grid), structure=np.ones((3, 3), bool))
    out = []
    for k in range(1, n + 1):
        cells = np.argwhere(labels == k)
        if len(cells) >= min_cells:
            out.append(cells)
    return out


def choose_frontier(grid, field, blacklist=(), size_weight=0.04, black_radius=0.5, min_clear=0.0):
    """Best (goal cell, cluster centroid xy, score) or None.

    The goal is the reachable cell nearest the frontier with at least `min_clear`
    metres to any obstacle (room to turn around there)."""
    clear = grid.obstacle_distance() if min_clear > 0 else None
    best = None
    for cells in frontier_clusters(grid):
        if clear is not None:                 # stand back from the frontier where there is room
            r0, c0 = cells.min(axis=0) - 12
            r1, c1 = cells.max(axis=0) + 12
            rr, cc = np.mgrid[max(r0, 0):min(r1, grid.shape[0]), max(c0, 0):min(c1, grid.shape[1])]
            ok = np.isfinite(field.cum[rr, cc]) & (clear[rr, cc] >= min_clear)
            cand = np.stack([rr[ok], cc[ok]], 1)
            if not len(cand):
                continue
            dmin = np.min(np.linalg.norm(cand[:, None, :] - cells[None, ::4, :], axis=2), axis=1)
            cand = cand[dmin <= 12]           # within 0.6 m of the frontier
            if not len(cand):
                continue
            rc = cand
        else:
            reach = np.isfinite(field.cum[cells[:, 0], cells[:, 1]])
            if not reach.any():
                continue
            rc = cells[reach]
        dist = field.cum[rc[:, 0], rc[:, 1]] * grid.res
        # aim at the reachable frontier cell nearest the cluster's middle
        centroid = grid.to_xy(*cells.mean(axis=0))
        xy = grid.to_xy(rc[:, 0], rc[:, 1])
        k = int(np.argmin(np.linalg.norm(xy - centroid, axis=1) + 0.3 * dist))
        goal = (int(rc[k, 0]), int(rc[k, 1]))
        gxy = grid.to_xy(*goal)
        if any(np.linalg.norm(gxy - np.asarray(b)) < black_radius for b in blacklist):
            continue
        score = dist[k] - size_weight * len(cells)
        if best is None or score < best[2]:
            best = (goal, gxy, score)
    return best
