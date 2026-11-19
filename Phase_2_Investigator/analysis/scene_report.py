"""From measured stains to a scene report (C5): pattern groups, mechanism, weapon category, area of origin.

    rep = analyse(observations, zone_of=..., gt=None)    # observations: analysis.closeup.merge_observations(...)
    print(rep["text"])

Steps (all Phase 2's existing analysis, now fed by the dog's own close-ups instead of ground truth):
1. the usable stains are grouped (single linkage, 0.3 m: `analysis.bpa.measure.group_observations`);
2. each group of >= `min_group` stains gets the interpretable pattern features and the mechanism
   classifier (`analysis/classifier`, gradient boosting trained on synthetic spatter). The explanation
   says which weapon categories the pattern is *consistent with* (never an identification);
3. impact / projected groups with >= `min_origin` stains get an area of origin: straight-line least squares,
   then gravity-corrected back-tracing with the mechanism's impact-speed prior, plus a 95 % ellipsoid
   (`analysis.bpa.origin`);
4. with ground truth (simulation): each group's true mechanism is the majority event of its matched stains,
   and the origin error is measured against that event's origin.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from analysis.bpa.measure import group_observations
from analysis.bpa.origin import MODEL_SD, SPEED_PRIORS, gravity_corrected_origin, straight_line_origin
from analysis.classifier.inference import explain
from analysis.classifier.model import DEFAULT_MODEL, MechanismClassifier


def load_classifier(log=print):
    """Phase 2's mechanism classifier (gitignored model; trained once if missing, ~2 min). A variant trained
    on degraded "close-up-like" measurements was tried and did worse on rendered close-ups, so it was dropped."""
    if not DEFAULT_MODEL.exists():
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        log("[analysis] training the mechanism classifier once (gitignored model, ~2 min)...")
        subprocess.run([sys.executable, str(root / "training/train_mechanism_classifier.py")], check=True,
                       cwd=root, stdout=subprocess.DEVNULL)
    return MechanismClassifier.load(DEFAULT_MODEL)


def _truth_event(group, gt, tol=0.006):
    st = [s for s in gt["stains"] if s.get("kind", "parent") == "parent"]
    P = np.array([s["center_world"] for s in st])
    ev = []
    for o in group:
        d = np.linalg.norm(P - o.position, axis=1)
        k = int(np.argmin(d))
        if d[k] <= max(tol, 0.6 * st[k]["length"]):
            ev.append(st[k]["event"])
    if not ev:
        return None, 0.0
    e, n = Counter(ev).most_common(1)[0]
    return e, n / len(group)


def _into_room(group, centre, step=0.4):
    """Where to look up a group's zone: walls are not in any zone, so step 0.4 m off the surface."""
    n = np.mean([o.plane.normal for o in group], axis=0)
    n = n / max(np.linalg.norm(n), 1e-9)
    return centre + step * n if abs(n[2]) < 0.8 else centre


def analyse(observations, zone_of=None, gt=None, min_group=3, min_origin=6, rng=None, log=print):
    rng = rng or np.random.default_rng(0)
    usable = [o for o in observations if o.usable]
    # group every measured stain, usable or not, exactly as the classifier's training data was grouped:
    # round stains (no measurable direction) and the usable fraction are features of the pattern
    groups = [g for g in group_observations(list(observations), link=0.3) if sum(o.usable for o in g) >= min_group]
    groups.sort(key=len, reverse=True)
    clf = load_classifier(log)
    events = {e["index"]: e for e in gt["events"]} if gt else {}
    out = []
    for i, g in enumerate(groups):
        gu = [o for o in g if o.usable]
        mech, probs, feats = clf.predict_group(g)
        ex = explain(probs, feats)
        P = np.array([o.position for o in g])
        c = P.mean(0)
        kinds = Counter(o.surface_kind for o in g).most_common(1)[0][0]
        item = {"id": f"g{i}", "n_stains": len(g), "n_usable": len(gu), "centre": np.round(c, 3).tolist(), "surface": kinds,
                "zone": zone_of(_into_room(g, c)) if zone_of else None, "mechanism": mech,
                "mechanism_confidence": round(float(probs[mech]), 3),
                "weapon_categories": ex["weapon_categories"], "text": ex["text"],
                "alpha_median_deg": round(float(np.median([o.alpha_deg for o in gu])), 1)}
        if mech in ("impact", "projected") and len(gu) >= min_origin:
            ms = [o.measured() for o in gu]
            sl = straight_line_origin(ms, rng=rng)
            gc = gravity_corrected_origin(ms, start=sl.point, speed_prior=SPEED_PRIORS[mech], rng=rng,
                                          model_sd=MODEL_SD[mech])
            item["origin"] = {"gravity_corrected": np.round(gc.point, 3).tolist(),
                              "straight_line": np.round(sl.point, 3).tolist(),
                              "ellipsoid_95_semi_axes_m": np.round(gc.semi_axes, 3).tolist(),
                              "height_m": round(float(gc.point[2]), 2)}
        if gt:
            e, purity = _truth_event(gu, gt)
            item["truth"] = {"event": e, "purity": round(purity, 2)}
            if e is not None:
                te = events[e]
                item["truth"]["mechanism"] = te["mechanism"]
                item["truth"]["mechanism_correct"] = te["mechanism"] == mech
                if "origin" in item:
                    true = np.array(te.get("origin_centroid") or te["origin"])
                    item["truth"]["origin_error_m"] = round(float(np.linalg.norm(np.array(item["origin"]["gravity_corrected"]) - true)), 3)
                    item["truth"]["origin_error_straight_m"] = round(float(np.linalg.norm(np.array(item["origin"]["straight_line"]) - true)), 3)
        out.append(item)
    rep = {"stains_measured": len(observations), "stains_usable": len(usable), "groups": out}
    if gt:
        scored = [g for g in out if g.get("truth", {}).get("event") is not None]
        rep["mechanism_correct"] = f"{sum(g['truth']['mechanism_correct'] for g in scored)}/{len(scored)}"
        errs = [g["truth"]["origin_error_m"] for g in scored if "origin_error_m" in g["truth"]]
        rep["origin_error_median_m"] = round(float(np.median(errs)), 3) if errs else None
        rep["true_events"] = [{"index": e["index"], "mechanism": e["mechanism"], "weapon_category": gt.get("weapon_category"),
                               "origin": np.round(e.get("origin_centroid") or e["origin"], 2).tolist()} for e in gt["events"]]
    rep["text"] = report_text(out)
    return rep


def report_text(groups):
    lines = ["# Scene report (generated by the dog)", ""]
    if not groups:
        return "\n".join(lines + ["No stain group large enough to analyse."])
    by_zone = {}
    for g in groups:
        by_zone.setdefault(g["zone"] or "unknown zone", []).append(g)
    for z, gs in by_zone.items():
        lines.append(f"## {z}")
        for g in gs:
            s = (f"- **{g['id']}**: {g['n_stains']} stains on the {g['surface']} around "
                 f"({g['centre'][0]:.2f}, {g['centre'][1]:.2f}, {g['centre'][2]:.2f}) m; median impact angle "
                 f"{g['alpha_median_deg']} deg. {g['text']}")
            if "origin" in g:
                o = g["origin"]
                s += (f" Estimated area of origin ({', '.join(f'{v:.2f}' for v in o['gravity_corrected'])}) m, "
                      f"about {o['height_m']:.2f} m above the floor (95 % ellipsoid up to "
                      f"{max(o['ellipsoid_95_semi_axes_m']):.2f} m).")
            lines.append(s)
        lines.append("")
    return "\n".join(lines)
