"""Mechanism -> weapon category, with confidence and a plain-language explanation.

Conclusions are always phrased as "consistent with", with wording that
follows the reliability notes in sim_env/config/weapon_mapping.yaml.
"""
from __future__ import annotations

from pathlib import Path

import yaml

MAPPING_PATH = Path(__file__).resolve().parents[2] / "sim_env" / "config" / "weapon_mapping.yaml"


def load_mapping(path=MAPPING_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def weapon_probabilities(mech_probs, mapping=None):
    mapping = mapping or load_mapping()
    out = {c: 0.0 for c in mapping["categories"]}
    for m, p in mech_probs.items():
        for c, w in mapping["mechanisms"][m]["weights"].items():
            out[c] += p * w
    tot = sum(out.values()) or 1.0
    return {c: v / tot for c, v in out.items()}


def confidence_word(p):
    return "high" if p >= 0.8 else "moderate" if p >= 0.55 else "low"


def evidence_lines(mech, f):
    """Short bullet points naming the measurements behind a mechanism."""
    n = int(round(2.718281828 ** f["log_n"]))
    lines = [f"{n} stains, median width {f['width_med_mm']:.1f} mm, median impact angle {f['alpha_med']:.0f} deg"]
    if mech == "impact":
        lines.append(f"{100 * f['convergence_frac']:.0f}% of measurable stains trace back to one common area")
        lines.append("mostly small stains (medium-velocity size range)")
    elif mech == "castoff":
        lines.append(f"stains lie along a line (linearity {f['linearity']:.2f}) about {f['extent_m']:.1f} m long")
        lines.append(f"directions run along that line ({f['along_axis']:.2f}, 1 = all parallel to it)")
        if f["frac_ceiling"] > 0:
            lines.append(f"{100 * f['frac_ceiling']:.0f}% of the stains are on the ceiling")
    elif mech == "projected":
        lines.append(f"large stains (up to {f['width_max_mm']:.0f} mm) in an extended band")
        lines.append("pattern consistent with blood ejected under pressure in pulses")
    elif mech == "drip":
        lines.append(f"{100 * f['frac_round']:.0f}% of stains are near-circular (blood falling vertically)")
        lines.append(f"similar sizes (width variation {100 * f['width_cv']:.0f}%) around {f['width_med_mm']:.0f} mm")
        if f["frac_round"] < 0.6:
            lines.append("elongated drops in a line suggest the source was moving")
    elif mech == "transfer":
        lines.append(f"one large contact area of about {f['max_area_cm2']:.0f} cm2 rather than droplets")
    return lines


def explain(mech_probs, feats, mapping=None):
    """Structured result for the GUI / report."""
    mapping = mapping or load_mapping()
    mech = max(mech_probs, key=mech_probs.get)
    p = mech_probs[mech]
    info = mapping["mechanisms"][mech]
    wording = mapping["reliability_wording"].get(info["reliability"].split(",")[0].split("-for")[0],
                                                 "is consistent with")
    wp = weapon_probabilities(mech_probs, mapping)
    ranked = sorted(wp.items(), key=lambda kv: -kv[1])
    cats = [mapping["categories"][c]["label"] for c, v in ranked if v >= 0.25] or \
           [mapping["categories"][ranked[0][0]]["label"]]
    text = (f"This stain group {wording} {info['label']} "
            f"(classifier confidence {confidence_word(p)}, {100 * p:.0f}%). "
            f"Weapon categories consistent with this: {', '.join(cats)}. "
            f"These are inferences from the stain pattern, not identifications.")
    return {"mechanism": mech, "mechanism_label": info["label"], "mechanism_confidence": p,
            "confidence_word": confidence_word(p), "mechanism_probs": mech_probs,
            "weapon_probs": wp, "weapon_categories": cats, "reliability": info["reliability"],
            "notes": info["notes"], "evidence": evidence_lines(mech, feats), "text": text}
