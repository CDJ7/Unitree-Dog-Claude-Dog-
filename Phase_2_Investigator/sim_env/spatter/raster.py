"""Draw stains as pixel masks in a surface's (u, v) frame.

Two uses:
  * `stain_mask` renders a single stain on its own small patch. The BPA tests
    and the idealised evaluation fit ellipses to these (perfect segmentation).
  * `decal_patch` renders every stain of a region into one RGBA image, the
    texture that later goes onto a thin visual-only quad in the MuJoCo scene.

Pixel (row i, col j) covers u = u0 + (j + 0.5) * res, v = v0 + (i + 0.5) * res,
so rows run along +v. Flip vertically before saving if you want +v up.
"""
from __future__ import annotations

import numpy as np

BLOOD_RGB = np.array([110, 8, 12], np.uint8)        # dried-blood red, flat and non-graphic


def _shape_field(stain, uu, vv):
    """Boolean coverage of one stain (ellipse + tapered tail) on a grid of u, v points."""
    cu, cv = stain["center_uv"]
    g = np.radians(stain["gamma_deg"])
    fu, fv = np.cos(g), np.sin(g)
    du, dv = uu - cu, vv - cv
    s = du * fu + dv * fv                 # along travel
    q = -du * fv + dv * fu                # across
    a = stain["length"] / 2
    b = stain["width"] / 2
    inside = (s / a) ** 2 + (q / b) ** 2 <= 1.0
    tl = stain.get("tail_length", 0.0)
    if tl > 0:
        # tail: from the downstream end, narrowing from 0.35 W to a point
        t = (s - a * 0.85) / (tl + a * 0.15)
        half = 0.35 * b * np.clip(1 - t, 0, 1) ** 1.3
        inside |= (t >= 0) & (t <= 1) & (np.abs(q) <= half)
    return inside


def stain_mask(stain, res=1e-4, margin=2):
    """Mask of one stain on a patch just big enough for it.

    Returns (mask[H, W] bool, (u0, v0), res): the uv of the patch's lower-left corner.
    """
    ext = stain["length"] / 2 + stain.get("tail_length", 0.0) + margin * res
    cu, cv = stain["center_uv"]
    u0, v0 = cu - ext, cv - ext
    n = int(np.ceil(2 * ext / res))
    c = (np.arange(n) + 0.5) * res
    uu, vv = np.meshgrid(u0 + c, v0 + c)
    return _shape_field(stain, uu, vv), (u0, v0), res


def mask_points(mask, origin_uv, res):
    """uv coordinates of the centres of the True pixels of a mask."""
    i, j = np.nonzero(mask)
    return np.stack([origin_uv[0] + (j + 0.5) * res, origin_uv[1] + (i + 0.5) * res], axis=1)


def smear_field(smear, uu, vv, rng=None):
    """Coverage of a smear: a band along its path, breaking into streaks at the end."""
    path = np.asarray(smear["path_uv"])
    w = smear["width"]
    cover = np.zeros(uu.shape, bool)
    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = seg_len.sum()
    acc = 0.0
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        d = b - a
        L = np.linalg.norm(d)
        if L == 0:
            continue
        d /= L
        du, dv = uu - a[0], vv - a[1]
        s = du * d[0] + dv * d[1]
        q = -du * d[1] + dv * d[0]
        frac = (acc + np.clip(s, 0, L)) / total
        on = (s >= 0) & (s <= L) & (np.abs(q) <= w / 2)
        feather = frac > 1 - smear["feather_frac"]
        # streaks: in the feathered end only some stripes across the width remain
        stripes = np.sin(q / w * 2 * np.pi * 7) > (2 * (frac - (1 - smear["feather_frac"])) / smear["feather_frac"] - 1)
        cover |= on & (~feather | stripes)
        acc += L
    return cover


def decal_patch(stains, smears, bounds_uv, res=5e-4, rgb=BLOOD_RGB):
    """RGBA uint8 image of every stain/smear inside bounds_uv = (u_min, v_min, u_max, v_max)."""
    u_min, v_min, u_max, v_max = bounds_uv
    w = int(np.ceil((u_max - u_min) / res))
    h = int(np.ceil((v_max - v_min) / res))
    img = np.zeros((h, w, 4), np.uint8)
    cov = np.zeros((h, w), bool)
    for st in stains:
        ext = st["length"] / 2 + st.get("tail_length", 0.0)
        cu, cv = st["center_uv"]
        j0, j1 = int((cu - ext - u_min) / res) - 1, int((cu + ext - u_min) / res) + 2
        i0, i1 = int((cv - ext - v_min) / res) - 1, int((cv + ext - v_min) / res) + 2
        j0, i0 = max(j0, 0), max(i0, 0)
        j1, i1 = min(j1, w), min(i1, h)
        if j1 <= j0 or i1 <= i0:
            continue
        uu, vv = np.meshgrid(u_min + (np.arange(j0, j1) + 0.5) * res, v_min + (np.arange(i0, i1) + 0.5) * res)
        cov[i0:i1, j0:j1] |= _shape_field(st, uu, vv)
    if smears:
        uu, vv = np.meshgrid(u_min + (np.arange(w) + 0.5) * res, v_min + (np.arange(h) + 0.5) * res)
        for sm in smears:
            cov |= smear_field(sm, uu, vv)
    img[cov, :3] = rgb
    img[cov, 3] = 255
    return img


def stain_bounds(stains, smears=(), pad=0.02):
    """uv bounding box of a set of stains on one surface."""
    pts = []
    for st in stains:
        e = st["length"] / 2 + st.get("tail_length", 0.0)
        cu, cv = st["center_uv"]
        pts += [(cu - e, cv - e), (cu + e, cv + e)]
    for sm in smears:
        for u, v in sm["path_uv"]:
            pts += [(u - sm["width"], v - sm["width"]), (u + sm["width"], v + sm["width"])]
    pts = np.array(pts)
    return (pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad)
