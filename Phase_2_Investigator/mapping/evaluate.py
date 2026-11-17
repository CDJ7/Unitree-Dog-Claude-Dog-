"""Score a zone map against ground-truth zone polygons (simulation only)."""
from __future__ import annotations

import numpy as np
import yaml


def load_polygons(path):
    cfg = yaml.safe_load(open(path))
    return [z["name"] for z in cfg["zones"]], [np.asarray(z["polygon"], float) for z in cfg["zones"]], cfg["environment"]


def _inside(poly, x, y):
    """Even-odd point-in-polygon for arrays x, y."""
    ins = np.zeros(x.shape, bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        ins ^= cond & (x < xin)
    return ins


def true_zone_grid(grid, names, polys):
    rr, cc = np.indices(grid.shape)
    xy = grid.to_xy(rr, cc)
    out = np.full(grid.shape, -1)
    for k, poly in enumerate(polys):
        out[(out < 0) & _inside(poly, xy[..., 0], xy[..., 1])] = k
    return out


def score(zone_map, gt_path):
    """Cell accuracy of the zone map over free cells inside a true zone, per zone and overall, and
    whether each zone's goal point lies in the right zone."""
    names, polys, env = load_polygons(gt_path)
    g = zone_map.grid
    lab, _ = zone_map.labels()
    idx = [zone_map.classes.index(n) if n in zone_map.classes else -2 for n in names]
    truth = true_zone_grid(g, names, polys)
    m = (lab >= 0) & (truth >= 0)
    true_cls = np.where(truth >= 0, np.array(idx)[np.maximum(truth, 0)], -3)
    per = {}
    for k, n in enumerate(names):
        mk = m & (truth == k)
        per[n] = {"cells": int(mk.sum()), "area_m2": round(float(mk.sum()) * g.res ** 2, 2),
                  "accuracy": round(float((lab[mk] == idx[k]).mean()), 4) if mk.any() else None}
    goals = {}
    for n, r in zone_map.regions().items():
        x, y = r["goal"]
        t = [names[k] for k, p in enumerate(polys) if _inside(p, np.array([x]), np.array([y]))[0]]
        goals[n] = {"goal": r["goal"], "true_zone_at_goal": t[0] if t else None, "correct": bool(t and t[0] == n)}
    return {"environment": env, "mapped_free_m2_in_zones": round(float(m.sum()) * g.res ** 2, 2),
            "cell_accuracy": round(float((lab[m] == true_cls[m]).mean()), 4) if m.any() else None,
            "per_zone": per, "goals": goals,
            "goals_correct": f"{sum(v['correct'] for v in goals.values())}/{len(goals)}"}
