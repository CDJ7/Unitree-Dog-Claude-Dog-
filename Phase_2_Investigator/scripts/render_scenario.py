#!/usr/bin/env python3
"""Overview sheet of a scenario: top-down, event area, close-ups of the two largest stain clusters.

    python scripts/render_scenario.py --scenario 1                # -> docs/media/a3_<id>.jpg
    python scripts/render_scenario.py --all
    python scripts/render_scenario.py --scenario 5 --rebuild     # regenerate spatter first
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401  (first: picks the offscreen GL backend before mujoco loads)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from sim_env.scenario_scene import load_scenario_config, open_scenario  # noqa: E402
from sim_env.scenarios import scenario_paths  # noqa: E402


def free_cam(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def facing(normal):
    """Azimuth (deg) of a camera looking at a surface with this outward normal."""
    n = np.asarray(normal, float)
    if abs(n[2]) > 0.9:                          # floor: look from the robot side, steeply down
        return 90.0
    return float(np.degrees(np.arctan2(-n[1], -n[0])))


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text((6, 5), text, fill=(255, 255, 255))
    return np.asarray(im)


def sheet(scn, out, rebuild=False):
    sim, gt = open_scenario(scn, force=rebuild)
    try:
        W, H = 640, 480
        spawn = np.asarray(scn["spawn"][:2], float)
        pts = [spawn] + [np.asarray(c["center"][:2]) for c in gt["clusters"]] + \
              [np.asarray(m["position"][:2]) for m in gt["mannequins"]]
        lo, hi = np.min(pts, axis=0) - 1.2, np.max(pts, axis=0) + 1.2
        centre = (lo + hi) / 2
        span = max(hi - lo) * 1.05
        top = sim.render_view(W, H, camera=free_cam([*centre, 0], span * 1.25, 90, -89.9))
        ev = gt["events"][gt["events"][0]["index"]]
        o = np.asarray(ev.get("origin") or gt["clusters"][0]["center"], float)
        c0 = gt["clusters"][0]
        az = facing(c0["normal"])
        area = sim.render_view(W, H, camera=free_cam([o[0], o[1], 0.7], 3.2, az + 25, -25))
        views = [label(top, f"{gt['scenario_id']}: top-down (robot spawn at {spawn.round(1).tolist()})"),
                 label(area, f"event area: {ev['mechanism']} ({gt['weapon_category']}, {gt.get('weapon')})")]
        for c in gt["clusters"][:2]:
            el = -70 if c["surface_kind"] == "floor" else -10
            dist = float(np.clip(c["radius"] * 2.6, 0.35, 1.6))
            img = sim.render_view(W, H, camera=free_cam(c["center"], dist, facing(c["normal"]), el))
            views.append(label(img, f"cluster {c['id']}: {c['n_stains']} stains on {c['surface_kind']} "
                                    f"({', '.join(c['mechanisms'])}), {dist:.2f} m"))
        while len(views) < 4:
            views.append(np.zeros((H, W, 3), np.uint8))
        grid = np.vstack([np.hstack(views[:2]), np.hstack(views[2:4])])
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(grid).save(out, quality=85)
        print(f"wrote {out}  ({len(gt['stains'])} stains, {len(gt['clusters'])} clusters)")
    finally:
        sim.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--out-dir", default=str(ROOT / "docs" / "media"))
    a = ap.parse_args()
    specs = [str(i + 1) for i in range(len(scenario_paths()))] if a.all else [a.scenario or "1"]
    for spec in specs:
        scn = load_scenario_config(spec)
        sheet(scn, Path(a.out_dir) / f"a3_{scn['id']}.jpg", a.rebuild)


if __name__ == "__main__":
    main()
