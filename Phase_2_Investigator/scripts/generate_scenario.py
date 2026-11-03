#!/usr/bin/env python3
"""Generate a scenario's spatter: ground-truth JSON, decal textures and a 3D preview.

    python scripts/generate_scenario.py --scenario 1
    python scripts/generate_scenario.py --config sim_env/scenarios/analytic/s2_kitchen_edged.yaml
    python scripts/generate_scenario.py --random 42 --out outputs/random42

Outputs (default outputs/<scenario id>/, gitignored):
  ground_truth.json         every droplet, stain, smear and event parameter
  decals/<surface>.png      RGBA stain texture per surface with stains (+v up)
  decals/index.json         where each decal sits on its surface (uv bounds, metres)
  preview.png               3D view: stains by event, origins, swing paths
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from sim_env.scenarios import analytic_paths as scenario_paths  # noqa: E402
from sim_env.scenarios.random_scenario import random_scenario  # noqa: E402
from sim_env.spatter import generate, load_scenario, save_ground_truth, summary  # noqa: E402
from sim_env.spatter.raster import decal_patch, stain_bounds  # noqa: E402


def write_decals(gt, out_dir, res):
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    surf_ids = sorted({s["surface_id"] for s in gt["stains"]} | {s["surface_id"] for s in gt["smears"]})
    for sid in surf_ids:
        st = [s for s in gt["stains"] if s["surface_id"] == sid]
        sm = [s for s in gt["smears"] if s["surface_id"] == sid]
        bounds = stain_bounds(st, sm)
        img = decal_patch(st, sm, bounds, res=res)
        Image.fromarray(img[::-1]).save(out_dir / f"{sid}.png")        # flip so +v is up
        index[sid] = {"file": f"{sid}.png", "uv_bounds": list(bounds), "res_m_per_px": res,
                      "size_px": [img.shape[1], img.shape[0]]}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index


def preview(gt, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    W, D, H = gt["room"]["size"]
    o = gt["room"].get("origin", [0, 0, 0])
    for x in (0, W):
        for y in (0, D):
            ax.plot([o[0] + x] * 2, [o[1] + y] * 2, [0, H], color="0.8", lw=0.8)
    colors = plt.cm.tab10.colors
    for ev in gt["events"]:
        c = colors[ev["index"] % 10]
        pts = np.array([s["center_world"] for s in gt["stains"] if s["event"] == ev["index"] and s["kind"] == "parent"])
        if len(pts):
            ax.scatter(*pts.T, s=6, color=c, label=f"{ev['index']}: {ev['mechanism']}")
        if ev.get("origin") is not None:
            ax.scatter(*np.array(ev["origin"]), marker="*", s=180, color=c, edgecolor="k")
        if ev.get("swing_path"):
            ax.plot(*np.array(ev["swing_path"]).T, color=c, lw=2)
    for sm in gt["smears"]:
        ax.plot(*np.array(sm["path_world"]).T, color="darkred", lw=4)
    for m in gt["mannequins"]:
        x, y = m["position"]
        ax.plot([x, x], [y, y], [0, m.get("height", 1.7) * (0.35 if m.get("posture") == "lying" else 1)],
                color="0.3", lw=3)
    ax.set_xlim(o[0], o[0] + W), ax.set_ylim(o[1], o[1] + D), ax.set_zlim(0, H)
    ax.set_box_aspect((W, D, H))
    ax.set_xlabel("x (m)"), ax.set_ylabel("y (m)"), ax.set_zlabel("z (m)")
    ax.set_title(f"{gt['scenario_id']}: stains by event (stars = true origins)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scenario", type=int, help="number of a built-in scenario (1-%d)" % len(scenario_paths()))
    g.add_argument("--config", help="path to a scenario YAML")
    g.add_argument("--random", type=int, metavar="SEED", help="random scenario with this seed")
    ap.add_argument("--seed", type=int, help="override the scenario's seed")
    ap.add_argument("--out", help="output folder (default outputs/<scenario id>)")
    ap.add_argument("--decal-res", type=float, default=1e-3, help="decal texture metres per pixel (default 1 mm)")
    ap.add_argument("--no-decals", action="store_true")
    a = ap.parse_args()

    if a.scenario is not None:
        scn = load_scenario(scenario_paths()[a.scenario - 1])
    elif a.config:
        scn = load_scenario(a.config)
    else:
        scn = random_scenario(a.random)
    gt = generate(scn, seed=a.seed)
    out = Path(a.out) if a.out else ROOT / "outputs" / gt["scenario_id"]
    save_ground_truth(gt, out / "ground_truth.json")
    if not a.no_decals:
        write_decals(gt, out / "decals", a.decal_res)
    preview(gt, out / "preview.png")
    print(summary(gt))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
