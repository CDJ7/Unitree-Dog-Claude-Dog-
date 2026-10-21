"""Simulated measurement of ground-truth stains (no simulator or camera needed).

Every stain is rasterised at `res` metres per pixel (what a close-up camera
would resolve on the surface), fitted with the same ellipse code the live
pipeline uses, and turned into an `Observation`. Stains narrower than
`min_px` pixels are lost (not resolvable), and a random fraction is dropped
to mimic occlusion and missed detections. Smears become one large
observation each.

This is the "idealised segmentation" stage: SAM and RF-DETR errors are not
modelled here; they are measured separately once those models run.
"""
from __future__ import annotations

import numpy as np

from analysis.bpa.measure import from_uv_points
from analysis.bpa.projection import Plane
from sim_env.spatter.raster import mask_points, smear_field, stain_mask


def plane_of(surface):
    return Plane(np.array(surface["center"]), np.array(surface["normal"]), np.array(surface["u"]),
                 np.array(surface["v"]))


def observe(gt, res=2.5e-4, min_px=4, drop=0.1, rng=None, include_satellites=True, smear_res=2e-3):
    """Observations of every stain and smear in a ground-truth dict.

    Returns (observations, truth) where truth[i] is the stain/smear dict the
    observation came from (for scoring).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    surf = {s["id"]: s for s in gt["surfaces"]}
    planes = {k: plane_of(v) for k, v in surf.items()}
    obs, truth = [], []
    for st in gt["stains"]:
        if not include_satellites and st["kind"] != "parent":
            continue
        if st["width"] < min_px * res or rng.uniform() < drop:
            continue
        mask, o, r = stain_mask(st, res=res)
        pts = mask_points(mask, o, r)
        if len(pts) < 6:
            continue
        try:
            ob = from_uv_points(st["id"], pts, r, planes[st["surface_id"]], st["surface_id"],
                                surf[st["surface_id"]]["kind"])
        except ValueError:
            continue
        ob.meta.update(event=st["event"], true_alpha=st["impact_angle_deg"], true_gamma=st["gamma_deg"],
                       kind=st["kind"])
        obs.append(ob)
        truth.append(st)
    for sm in gt["smears"]:
        path = np.asarray(sm["path_uv"])
        lo, hi = path.min(0) - sm["width"], path.max(0) + sm["width"]
        uu, vv = np.meshgrid(np.arange(lo[0], hi[0], smear_res), np.arange(lo[1], hi[1], smear_res))
        cov = smear_field(sm, uu, vv)
        pts = np.stack([uu[cov], vv[cov]], axis=1)
        if len(pts) < 6:
            continue
        ob = from_uv_points(sm["id"], pts, smear_res, planes[sm["surface_id"]], sm["surface_id"],
                            surf[sm["surface_id"]]["kind"])
        ob.usable = False
        ob.reason = "smear"
        ob.meta.update(event=sm["event"], kind="smear")
        obs.append(ob)
        truth.append(sm)
    return obs, truth


def label_group(group, gt):
    """Mechanism of the event that contributed most stains to a group."""
    events = [o.meta.get("event") for o in group]
    vals, counts = np.unique(events, return_counts=True)
    ev = int(vals[np.argmax(counts)])
    return gt["events"][ev]["mechanism"], ev, float(counts.max() / len(group))
