"""Evidence map (C3): stain detections from many camera frames -> evidence clusters on the dog's map.

Each detected stain region (perception/stain_detector.py) carries a world position and a surface normal
(from depth in simulation; on the real Go2 from the floor plane or the LiDAR). The evidence map:
- groups detections into clusters: a detection joins the nearest cluster within `link` m on a similar
  surface (normals within ~45 deg), else it starts a new one;
- keeps per cluster: position (weighted mean), normal, surface kind (floor / wall / raised = furniture),
  the number of frames it was seen in, mean and max detector probability, and the zone it lies in (from the
  zone map, when there is one);
- calls a cluster *confirmed* once it was seen in >= `min_views` frames with mean probability >=
  `min_prob`. Single-frame false alarms (a red cushion seen once at a bad angle) stay unconfirmed;
- gives the floor clusters' keep-out disk for planning (C4: never walk over evidence).
`score()` compares confirmed clusters with the ground-truth stain clusters (simulation only).
"""
from __future__ import annotations

import json

import numpy as np


class EvidenceMap:
    def __init__(self, link=0.35, min_views=2, min_prob=0.5, normal_dot=0.7):
        self.link, self.min_views, self.min_prob, self.normal_dot = link, min_views, min_prob, normal_dot
        self.clusters = []
        self.frames = 0

    @staticmethod
    def surface_kind(xyz, normal):
        nz = abs(float(normal[2]))
        if nz > 0.8:
            return "floor" if xyz[2] < 0.10 else "raised"
        if nz < 0.35:
            return "wall"
        return "slanted"

    def add_frame(self, detections, t=None, zone_of=None):
        """detections: regions with .xyz, .normal, .prob (and .feats). zone_of: callable(xy) -> zone name."""
        self.frames += 1
        for d in detections:
            if d.xyz is None or not np.all(np.isfinite(d.xyz)):
                continue
            p, n = np.asarray(d.xyz, float), np.asarray(d.normal if d.normal is not None else [0, 0, 1], float)
            area = float(np.exp(d.feats.get("log_area_m2", np.log(1e-4)))) if d.feats else 1e-4
            best, bd = None, self.link
            for c in self.clusters:
                dist = float(np.linalg.norm(c["xyz"] - p))
                if dist < bd and abs(float(c["normal"] @ n)) >= self.normal_dot:
                    best, bd = c, dist
            if best is None:
                best = {"id": f"e{len(self.clusters)}", "xyz": p.copy(), "normal": n / max(np.linalg.norm(n), 1e-9),
                        "w": 0.0, "views": 0, "probs": [], "area_m2": 0.0, "first_t": t, "_last_frame": -1}
                self.clusters.append(best)
            w = d.prob
            best["xyz"] = (best["xyz"] * best["w"] + p * w) / (best["w"] + w + 1e-12)
            nn = best["normal"] * best["w"] + n * w * np.sign(float(best["normal"] @ n) or 1.0)
            best["normal"] = nn / max(np.linalg.norm(nn), 1e-9)
            best["w"] += w
            best["probs"].append(float(d.prob))
            best["area_m2"] = max(best["area_m2"], area)
            if best["_last_frame"] != self.frames:
                best["views"] += 1
                best["_last_frame"] = self.frames
            best["last_t"] = t
            if zone_of is not None:
                best["zone"] = zone_of(best["xyz"][:2])
            if "confirmed_t" not in best and best["views"] >= self.min_views and np.mean(best["probs"]) >= self.min_prob:
                best["confirmed_t"] = t

    def confirmed(self):
        return [c for c in self.clusters if c["views"] >= self.min_views and np.mean(c["probs"]) >= self.min_prob]

    def summary(self, only_confirmed=False):
        out = []
        for c in (self.confirmed() if only_confirmed else self.clusters):
            out.append({"id": c["id"], "xyz": np.round(c["xyz"], 3).tolist(), "normal": np.round(c["normal"], 3).tolist(),
                        "surface": self.surface_kind(c["xyz"], c["normal"]), "views": c["views"],
                        "mean_prob": round(float(np.mean(c["probs"])), 3), "max_prob": round(float(np.max(c["probs"])), 3),
                        "area_m2_max": round(c["area_m2"], 4), "zone": c.get("zone"),
                        "confirmed": bool(c["views"] >= self.min_views and np.mean(c["probs"]) >= self.min_prob)})
        return out

    def floor_points(self, margin=0.15):
        """(xy, radius) of confirmed floor evidence, for the planner's keep-out layer."""
        out = []
        for c in self.confirmed():
            if self.surface_kind(c["xyz"], c["normal"]) == "floor":
                r = float(np.sqrt(max(c["area_m2"], 1e-4) / np.pi)) + margin
                out.append((c["xyz"][:2].copy(), min(r, 0.8)))
        return out

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"frames": self.frames, "link_m": self.link, "min_views": self.min_views, "min_prob": self.min_prob,
                       "clusters": self.summary()}, fh, indent=1)


def score(evidence, gt, match_m=0.6, min_stains=3):
    """Confirmed evidence clusters vs ground-truth clusters (>= min_stains stains): recall, precision, error."""
    truth = [c for c in gt.get("clusters", []) if c["n_stains"] >= min_stains]
    found = evidence.confirmed()
    used, errs, matched_truth = set(), [], set()
    for c in found:
        best, bd = None, match_m
        for k, t in enumerate(truth):
            dist = float(np.linalg.norm(np.asarray(t["center"]) - c["xyz"]))
            # a big cluster may be found as several pieces: distance to its bounding box counts too
            lo, hi = np.asarray(t["bbox_min"]) - 0.1, np.asarray(t["bbox_max"]) + 0.1
            inside = bool(np.all(c["xyz"] >= lo) and np.all(c["xyz"] <= hi))
            dd = 0.0 if inside else dist
            if dd < bd:
                best, bd = k, dd
        if best is not None:
            used.add(c["id"])
            matched_truth.add(best)
            errs.append(float(np.linalg.norm(np.asarray(truth[best]["center"]) - c["xyz"])))
    distractors = []
    for d in gt.get("distractors", []):
        cen = np.asarray(d["center"], float)
        if len(cen) == 2:
            cen = np.r_[cen, 0.4]
        hit = [c["id"] for c in found if c["id"] not in used and np.linalg.norm(cen - c["xyz"]) < 0.4]
        distractors.append({"id": d["id"], "flagged_as_evidence": bool(hit)})
    return {"truth_clusters": len(truth), "found_clusters": len(found),
            "truth_found": len(matched_truth), "recall": round(len(matched_truth) / max(len(truth), 1), 3),
            "precision": round(len(used) / max(len(found), 1), 3),
            "median_error_to_centre_m": round(float(np.median(errs)), 3) if errs else None,
            "truth_missed": [truth[k]["id"] for k in range(len(truth)) if k not in matched_truth],
            "distractors_flagged": sum(d["flagged_as_evidence"] for d in distractors), "distractors": len(distractors)}


def floor_stains_stepped_on(gt, path, half_length=0.35, half_width=0.155, times=False):
    """Ground-truth floor stains that the robot's body rectangle passed over along `path` [(t, x, y, yaw)].
    (The legs stand inside the body outline, so this counts every stain a foot could have touched.)
    times=True returns {stain id: (first time under the body, xy)}."""
    st = [s for s in gt["stains"] if s["center_world"][2] < 0.03]
    if not st or not len(path):
        return {} if times else []
    P = np.array([s["center_world"][:2] for s in st])
    first = np.full(len(P), np.nan)
    for t, x, y, yaw in path:
        d = P - (x, y)
        c, s_ = np.cos(yaw), np.sin(yaw)
        lx, ly = c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]
        new = (np.abs(lx) <= half_length) & (np.abs(ly) <= half_width) & np.isnan(first)
        first[new] = t
    idx = np.flatnonzero(~np.isnan(first))
    if times:
        return {st[i]["id"]: (float(first[i]), P[i]) for i in idx}
    return [st[i]["id"] for i in idx]


def stepped_after_found(evidence, stepped, near=0.5):
    """Stains first stepped on after a confirmed floor cluster within `near` m of them existed: the ones the
    keep-out layer was meant to protect. stepped: floor_stains_stepped_on(..., times=True)."""
    known = [(c["confirmed_t"], c["xyz"][:2]) for c in evidence.confirmed()
             if c.get("confirmed_t") is not None and evidence.surface_kind(c["xyz"], c["normal"]) == "floor"]
    out = []
    for sid, (t, xy) in stepped.items():
        if any(tc <= t and np.linalg.norm(xy - p) <= near for tc, p in known):
            out.append(sid)
    return out
