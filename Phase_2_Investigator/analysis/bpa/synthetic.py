"""Tiny pinhole renderer for stains on one planar surface (no MuJoCo needed).

Used by tests and the evaluation to exercise the full camera path
(mask + depth + intrinsics + pose -> measured angle) at realistic camera
resolutions and viewing angles, before the simulator is available.
"""
from __future__ import annotations

import numpy as np

from sim_env.spatter.raster import _shape_field


def render_plane_view(surface, stains, intr, T_world_cam, supersample=3, roi=None):
    """Render stains lying on `surface` (a geometry.Rect or its dict).

    Returns (labels[H, W] int: index into stains or -1, depth[H, W] float: z-depth
    of the surface, inf where the ray misses it). A pixel is labelled when most of
    its sub-samples fall inside a stain. roi = (x0, y0, x1, y1) limits the work.
    """
    if isinstance(surface, dict):
        c, n = np.array(surface["center"]), np.array(surface["normal"])
        u_ax, v_ax = np.array(surface["u"]), np.array(surface["v"])
        hu, hv = surface["half_u"], surface["half_v"]
    else:
        c, n, u_ax, v_ax, hu, hv = surface.center, surface.normal, surface.u, surface.v, surface.half_u, surface.half_v
    H, W = intr.height, intr.width
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, W, H)
    R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]

    def hit(px, py):
        rc = np.stack([(px - intr.cx) / intr.fx, (py - intr.cy) / intr.fy, np.ones_like(px)], axis=-1)
        rw = rc @ R.T
        den = rw @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((c - o) @ n) / den
        p = o + t[..., None] * rw
        d = p - c
        uu, vv = d @ u_ax, d @ v_ax
        ok = (t > 0) & (np.abs(uu) <= hu) & (np.abs(vv) <= hv)
        return uu, vv, t, ok                  # t = z-depth because rc has z = 1

    ys, xs = np.mgrid[y0:y1, x0:x1].astype(float)
    _, _, t, ok = hit(xs, ys)
    depth = np.full((H, W), np.inf)
    depth[y0:y1, x0:x1] = np.where(ok, t, np.inf)
    labels = np.full((H, W), -1, int)
    off = (np.arange(supersample) + 0.5) / supersample - 0.5
    for k, st in enumerate(stains):
        # project the stain's bounding circle to a pixel box
        ext = st["length"] / 2 + st.get("tail_length", 0.0)
        cw = c + st["center_uv"][0] * u_ax + st["center_uv"][1] * v_ax
        pc = (cw - o) @ R
        if pc[2] <= 0:
            continue
        px = intr.fx * pc[0] / pc[2] + intr.cx
        py = intr.fy * pc[1] / pc[2] + intr.cy
        r = intr.fx * ext / pc[2] * 3 + 2
        bx0, bx1 = int(max(px - r, x0)), int(min(px + r + 1, x1))
        by0, by1 = int(max(py - r, y0)), int(min(py + r + 1, y1))
        if bx1 <= bx0 or by1 <= by0:
            continue
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(float)
        cover = np.zeros(yy.shape)
        for dy in off:
            for dx in off:
                uu, vv, _, okk = hit(xx + dx, yy + dy)
                cover += okk & _shape_field(st, uu, vv)
        on = cover / supersample ** 2 >= 0.5
        sub = labels[by0:by1, bx0:bx1]
        sub[on & (sub < 0)] = k
    return labels, depth
