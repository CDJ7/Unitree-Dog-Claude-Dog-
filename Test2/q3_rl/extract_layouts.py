#!/usr/bin/env python3
"""Q3: export the real house-185 crime scenes into a light layout file for the RL environment.

    python q3_rl/extract_layouts.py                    # -> q3_rl/layouts/house185_layouts.npz + .json

For every scenario (S1, S2 and the random house-185 scenes listed in config/q3_env.yaml) the MuJoCo
scene is opened exactly as in Q2 (same furniture, props, mannequin, stains), and saved:
- the static collision footprint raster (5 cm cells, same rasteriser as the Q2 path planner), so the
  RL world's walls and furniture are the MuJoCo ones;
- the evidence: the ground-truth stain clusters (id, surface kind, centre, normal, radius, bbox).
The RL env then runs with plain numpy (no MuJoCo needed, so PPO trains fast on CPU / Colab).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

import go2sim.paths  # noqa: E402,F401  (puts Phase 2 on sys.path, picks a GL backend)
from go2sim.static_map import ORIGIN, RES, SIZE, build_grid  # noqa: E402

OUT = TEST2 / "q3_rl/layouts"


def scenario_spec(name):
    if name.startswith("r"):
        from sim_env.scenarios.random_sim import write_random
        return str(write_random(int(name[1:]), env="scenesmith_house_185"))
    return name[1:]          # "s1" -> "1"


def extract(name):
    from sim_env.scenario_scene import load_scenario_config, open_scenario
    scn = load_scenario_config(scenario_spec(name))
    sim, gt = open_scenario(scn, show_stains=False)
    try:
        grid, blocked, _ = build_grid(sim.model, sim.data, sim.robot_geoms)
    finally:
        sim.close() if hasattr(sim, "close") else None
    ev = []
    for c in gt["clusters"]:
        ev.append({"id": c["id"], "surface_kind": c["surface_kind"],
                   "center": [round(float(v), 4) for v in c["center"]],
                   "normal": [round(float(v), 4) for v in c["normal"]],
                   "radius": round(float(c["radius"]), 4),
                   "bbox_min": [round(float(v), 4) for v in c["bbox_min"]],
                   "bbox_max": [round(float(v), 4) for v in c["bbox_max"]],
                   "n_stains": int(c.get("n_stains", 0))})
    return blocked.astype(bool), {"name": name, "scenario_id": scn.get("id", name), "evidence": ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(TEST2 / "config/q3_env.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    names = cfg["layouts"]["train"] + cfg["layouts"]["val"] + cfg["layouts"]["test"]
    OUT.mkdir(parents=True, exist_ok=True)
    grids, meta = [], []
    for n in names:
        t0 = time.time()
        b, m = extract(n)
        grids.append(b)
        meta.append(m)
        kinds = {}
        for e in m["evidence"]:
            kinds[e["surface_kind"]] = kinds.get(e["surface_kind"], 0) + 1
        print(f"{n}: {len(m['evidence'])} clusters {kinds}, blocked {b.mean():.3f} ({time.time() - t0:.1f} s)", flush=True)
    np.savez_compressed(OUT / "house185_layouts.npz", blocked=np.stack(grids))
    (OUT / "house185_layouts.json").write_text(json.dumps(
        {"origin": list(ORIGIN), "size": list(SIZE), "res": RES, "names": names, "layouts": meta}, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
