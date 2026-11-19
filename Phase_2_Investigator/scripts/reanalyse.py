#!/usr/bin/env python3
"""Re-run the stain analysis of finished investigate.py runs, e.g. with another mechanism classifier.

    python scripts/reanalyse.py outputs/investigate/s1_living_room_blunt               # default classifier
    python scripts/reanalyse.py outputs/investigate/* --model ideal                    # Phase 2's original
    python scripts/reanalyse.py outputs/investigate/* --table docs/media/stage_c/missions.json

Needs the run's stains.pkl (the measured stains) and the scenario's ground truth. Writes analysis_<model>.json
and scene_report_<model>.md next to them; with --table, one row per run for the README.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--model", default="simclose", help="simclose (default) | ideal | path to a pickle")
    ap.add_argument("--table", help="write a summary table (JSON) of all runs here")
    a = ap.parse_args()
    from analysis.closeup import score_against_truth
    from analysis.scene_report import analyse
    rows = []
    for run in a.runs:
        run = Path(run)
        if not (run / "stains.pkl").exists():
            continue
        rep = json.loads((run / "report.json").read_text())
        scn = rep["scenario"].replace("_nokeepout", "")
        gt = json.loads((ROOT / "sim_env/build/scenarios" / scn / "ground_truth.json").read_text())
        ev = json.loads((run / "evidence.json").read_text())
        stains = pickle.loads((run / "stains.pkl").read_bytes())
        # zones: the evidence clusters' zones (the mission's zone map is not saved), nearest cluster
        import numpy as np
        cl = [(np.array(c["xyz"]), c.get("zone")) for c in ev["clusters"] if c.get("zone")]
        zone_of = (lambda p: min(cl, key=lambda c: np.linalg.norm(c[0] - p))[1]) if cl else None
        ana = analyse(stains, zone_of=zone_of, gt=gt, model=a.model)
        ana["measurement"] = score_against_truth(stains, gt)
        tag = Path(a.model).stem if a.model not in ("simclose", "ideal") else a.model
        (run / f"scene_report_{tag}.md").write_text(ana.pop("text"))
        (run / f"analysis_{tag}.json").write_text(json.dumps(ana, indent=1, default=float))
        s = rep["score"]
        rows.append({"run": run.name, "clusters_found": f"{s['truth_found']}/{s['truth_clusters']}",
                     "precision": s["precision"], "distractors_flagged": f"{s['distractors_flagged']}/{s['distractors']}",
                     "floor_stains_stepped_on": f"{rep['floor_stains_stepped_on']}/{rep['floor_stains_total']}",
                     "stepped_after_found": rep.get("floor_stains_stepped_on_after_found"),
                     "collisions": rep["n_collisions"], "closeups": sum(i.get("closeups", 0) for i in rep.get("inspected", [])),
                     "stains_measured": ana["stains_measured"],
                     "alpha_err_median_deg": ana["measurement"].get("alpha_abs_err_median_deg"),
                     "mechanism_correct": ana.get("mechanism_correct"),
                     "origin_error_median_m": ana.get("origin_error_median_m"),
                     "groups": [(g["n_stains"], g["surface"], g["mechanism"], g.get("truth", {}).get("mechanism"))
                                for g in ana["groups"]]})
        print(json.dumps(rows[-1]))
    if a.table:
        Path(a.table).write_text(json.dumps({"model": a.model, "runs": rows}, indent=1))


if __name__ == "__main__":
    main()
