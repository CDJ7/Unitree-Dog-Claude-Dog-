"""Group stains into spatially connected clusters: the "stain regions" a robot inspects.

Single linkage on each surface: stains closer than `link` metres belong to the
same cluster. Smears count as their path points. Each cluster reports its
centroid, the surface normal (facing the viewer), extent, stain count, and
which events/mechanisms contributed: these are the inspection targets (A4)
and the stain-region labels for the dataset (A6).
"""
from __future__ import annotations

import numpy as np


def _components(points, link):
    n = len(points)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        d = np.linalg.norm(points[i + 1:] - points[i], axis=1)
        for j in np.flatnonzero(d < link) + i + 1:
            a, b = find(i), find(j)
            if a != b:
                parent[b] = a
    return np.array([find(i) for i in range(n)])


def stain_clusters(gt, link=0.25, min_stains=3):
    surf = {s["id"]: s for s in gt["surfaces"]}
    mech = {e["index"]: e["mechanism"] for e in gt["events"]}
    items = []                                   # (surface, point, event, stain id or smear id)
    for st in gt["stains"]:
        items.append((st["surface_id"], st["center_world"], st["event"], st["id"]))
    for sm in gt["smears"]:
        for p in sm["path_world"]:
            items.append((sm["surface_id"], p, sm["event"], sm["id"]))
    clusters = []
    for sid in sorted({it[0] for it in items}):
        sel = [it for it in items if it[0] == sid]
        pts = np.array([it[1] for it in sel], float)
        labels = _components(pts, link)
        for lab in np.unique(labels):
            idx = np.flatnonzero(labels == lab)
            members = [sel[i] for i in idx]
            is_smear = any(m[3].endswith(tuple(f"smear{k}" for k in range(10))) for m in members)
            n_stains = len({m[3] for m in members})
            if n_stains < min_stains and not is_smear:
                continue
            p = pts[idx]
            events = sorted({int(m[2]) for m in members})
            c = p.mean(axis=0)
            clusters.append({
                "id": f"c{len(clusters)}", "surface_id": sid, "surface_kind": surf[sid]["kind"],
                "center": c.tolist(), "normal": surf[sid]["normal"],
                "radius": float(np.max(np.linalg.norm(p - c, axis=1))) if len(p) > 1 else 0.02,
                "bbox_min": p.min(axis=0).tolist(), "bbox_max": p.max(axis=0).tolist(),
                "n_stains": n_stains, "events": events,
                "mechanisms": sorted({mech[e] for e in events}),
                "stain_ids": sorted({m[3] for m in members}),
            })
    clusters.sort(key=lambda c: -c["n_stains"])
    for i, c in enumerate(clusters):
        c["id"] = f"c{i}"
    return clusters
