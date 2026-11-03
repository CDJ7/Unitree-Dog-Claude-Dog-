#!/usr/bin/env python3
"""Event playback (A5): reenact a scenario's event in the simulated scene.

    python scripts/playback.py --scenario 1                       # MP4 -> outputs/playback/<id>_ground_truth.mp4
    python scripts/playback.py --scenario 1 --viewer              # live in the MuJoCo viewer (loops)
    python scripts/playback.py --scenario 1 --from-descriptions   # rebuild from mechanism/origin/params only
                                                                  # (the path Stage B's analysis will use)
Options: --speed 0.25 (slow motion), --size 960x540, --fps 30, --gif, --snapshots (contact sheet PNG).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401  (first: picks the offscreen GL backend before mujoco loads)

import numpy as np  # noqa: E402

from sim_env.launch import add_sim_args, scenario_from_args  # noqa: E402
from sim_env.playback import EventPlayback, descriptions_from_ground_truth  # noqa: E402
from sim_env.scenario_scene import build_scenario  # noqa: E402


def main():
    ap = add_sim_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--from-descriptions", action="store_true",
                    help="drop the exact launches; synthesise droplets from each event's description")
    ap.add_argument("--speed", type=float, default=0.25)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="960x544")
    ap.add_argument("--gif", action="store_true", help="write a GIF instead of MP4")
    ap.add_argument("--snapshots", action="store_true", help="also save a contact sheet of 8 moments")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not (a.scenario or a.random is not None):
        a.scenario = "1"
    scn = scenario_from_args(a)
    gt = json.loads((build_scenario(scn, force=a.rebuild) / "ground_truth.json").read_text())
    if a.from_descriptions:
        events = descriptions_from_ground_truth(gt)
        for e in events:
            e.launches = []
            e.source = "analysis"
        pb = EventPlayback.from_descriptions(scn, gt, events, label="FROM DESCRIPTIONS")
        tag = "descriptions"
    else:
        pb = EventPlayback.from_ground_truth(scn, gt)
        tag = "ground_truth"
    try:
        P = pb.plan
        print(f"{scn['id']}: {len(P.droplets)} droplets, {len(P.swings)} swings, {len(P.walks)} walks, "
              f"{P.duration:.1f} s of event time")
        for t0, t1, cap in P.captions:
            print(f"  {t0:5.2f}-{t1:5.2f} s  {cap}")
        if a.viewer:
            pb.play_viewer(speed=a.speed)
            return
        out_dir = Path(a.out or ROOT / "outputs" / "playback")
        out_dir.mkdir(parents=True, exist_ok=True)
        w, h = (int(x) for x in a.size.split("x"))
        path = out_dir / f"{scn['id']}_{tag}.{'gif' if a.gif else 'mp4'}"
        pb.render_video(path, speed=a.speed, fps=a.fps, size=(w, h))
        print(f"wrote {path}")
        if a.snapshots:
            from PIL import Image
            strike = [s["t1"] for s in P.swings] + [d["release"] + 0.05 for d in P.droplets[:1]]
            ts = sorted(set(np.round(np.r_[np.linspace(0.6, P.duration - 0.1, 8 - len(strike[:2])), strike[:2]], 2)))
            while len(ts) < 8:                       # always a full 2x4 sheet
                ts = sorted(set(ts) | {round(float(np.random.uniform(0.6, P.duration)), 2)})
            ts = ts[:8]
            imgs = [pb.snapshot(t, (480, 270)) for t in ts]
            sheet = out_dir / f"{scn['id']}_{tag}_sheet.jpg"
            Image.fromarray(np.vstack([np.hstack(imgs[:4]), np.hstack(imgs[4:])])).save(sheet, quality=85)
            print(f"wrote {sheet}")
    finally:
        pb.close()


if __name__ == "__main__":
    main()
