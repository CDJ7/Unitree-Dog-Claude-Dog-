"""Map figure: occupancy, camera coverage, robot path, goals, ground-truth stains, collisions."""
from __future__ import annotations

import numpy as np


def map_image(grid, coverage=None):
    img = np.full(grid.shape + (3,), 0.62)
    free, occ = grid.free(), grid.occupied()
    img[free] = 1.0
    if coverage is not None:
        img[free & coverage.seen] = (0.80, 0.95, 0.80)
    img[occ] = 0.0
    if coverage is not None:
        img[occ & coverage.seen] = (0.05, 0.45, 0.10)
    return img


def plot_map(nav, gt=None, title="", path=None, ax=None):
    import matplotlib
    if ax is None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = nav.grid
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(10, 7.5))
    ext = [g.origin[0], g.origin[0] + g.shape[1] * g.res, g.origin[1], g.origin[1] + g.shape[0] * g.res]
    ax.imshow(map_image(g, nav.coverage), origin="lower", extent=ext, interpolation="nearest")
    p = np.array([(x, y) for _, x, y, _ in nav.log.path])
    if len(p):
        ax.plot(p[:, 0], p[:, 1], "-", color="tab:blue", lw=1.2, label="robot path")
        ax.plot(p[0, 0], p[0, 1], "o", color="tab:blue", ms=7)
    for t, kind, x, y, ok in nav.log.goals:
        ax.plot(x, y, "^" if kind == "frontier" else "s", color="tab:green" if ok else "tab:orange", ms=6)
    if gt:
        st = np.array([s["center_world"] for s in gt["stains"]])
        if len(st):
            ax.plot(st[:, 0], st[:, 1], ".", color="red", ms=2, label="stains (ground truth)")
        for c in gt.get("clusters", []):
            ax.annotate(c["id"], c["center"][:2], color="darkred", fontsize=7)
        for m in gt.get("mannequins", []):
            if m.get("present", True):
                ax.plot(*m["position"][:2], "P", color="purple", ms=9)
    for c in nav.log.collisions:
        ax.plot(c.position[0], c.position[1], "x", color="red", ms=14, mew=3)
    inspected = getattr(nav, "inspected", [])
    for item in inspected:
        for v in item.get("views", []):
            ax.plot(*v["robot_xy"], "D", color="magenta", ms=5)
    lo = p.min(axis=0) - 2.0 if len(p) else g.origin
    hi = p.max(axis=0) + 2.0 if len(p) else g.origin + 10
    if gt:
        pts = np.array([s["center_world"][:2] for s in gt["stains"]] or [[0, 0]])
        lo, hi = np.minimum(lo, pts.min(0) - 0.5), np.maximum(hi, pts.max(0) + 0.5)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=7)
    if own and path:
        fig.tight_layout()
        fig.savefig(path, dpi=90)
        plt.close(fig)
    return ax
