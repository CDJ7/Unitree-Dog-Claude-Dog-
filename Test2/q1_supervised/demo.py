#!/usr/bin/env python3
"""Q1 demo: the zone classifier running while the Go2 walks through the crime scene.

    python q1_supervised/demo.py                       # recorded held-out test run (first test run: run_000)
    python q1_supervised/demo.py --run run_031
    python q1_supervised/demo.py --live                # live in MuJoCo on r1854 (Q1's unseen test scene)
    python q1_supervised/demo.py --live --scenario 1   # live on scenario S1 (a training scene)

Recorded mode replays the test run's frames through the classifier (the run was never used in
training). Live mode opens the MuJoCo scene, walks a tour of all zones with the waypoint follower
and classifies the head-camera image every 0.2 s as it is rendered.
Both produce results/q1/demo_<name>.png (true vs predicted zone over time, raw and with 1 s causal
majority smoothing, plus the confidence) and demo_<name>.gif (frame, prediction, map).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from go2sim.zones import Zones  # noqa: E402
from q1_supervised.data import load_cfg, normalize  # noqa: E402
from q1_supervised.evaluate import load_model  # noqa: E402

EXTENT = (-0.5, 8.7, -0.5, 5.0)


class Classifier:
    def __init__(self, ckpt, zones):
        self.model, ck = load_model(ckpt, len(zones))
        self.h, self.w = ck["image_hw"]

    @torch.no_grad()
    def __call__(self, rgb):
        im = Image.fromarray(rgb).resize((self.w, self.h), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None].float() / 255.0
        return self.model(normalize(x)).softmax(1)[0].numpy()


def majority(pred, k=5):
    return np.array([np.bincount(pred[max(0, i - k + 1):i + 1]).argmax() for i in range(len(pred))])


def timeline_plot(df, zones, title, path):
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].step(df.t, df.true, where="post", color="k", lw=2.5, label="true zone (from dog position)")
    ax[0].plot(df.t, df.pred, ".", ms=4, color="C1", alpha=0.6, label="predicted (single frame)")
    ax[0].step(df.t, df.pred_smooth, where="post", color="C0", lw=1.2, label="predicted, 1 s majority")
    ax[0].set_yticks(range(len(zones)), zones.names)
    acc, accs = (df.true == df.pred).mean(), (df.true == df.pred_smooth).mean()
    ax[0].set_title(f"{title}: frame accuracy {acc:.3f}, smoothed {accs:.3f}")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(df.t, df.conf, color="C2", lw=0.8)
    wrong = df.true != df.pred
    ax[1].plot(df.t[wrong], df.conf[wrong], "x", color="red", ms=3, label="wrong")
    ax[1].set_ylabel("confidence")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return float(acc), float(accs)


def make_gif(images, df, zones, path, every=2, scale=0.8):
    bg_p = TEST2 / "results/q2/house185_topdown.png"
    bg = Image.open(bg_p).convert("RGB")
    H = images[0].height
    mw = int(bg.width * H / bg.height)
    bg = bg.resize((mw, H))

    def px(x, y):
        return ((x - EXTENT[0]) / (EXTENT[1] - EXTENT[0]) * mw, (EXTENT[3] - y) / (EXTENT[3] - EXTENT[2]) * H)

    out, trail = [], []
    for k in range(0, len(df)):
        r = df.iloc[k]
        trail.append(px(r.base_x, r.base_y))
        if k % every:
            continue
        canvas = Image.new("RGB", (images[k].width + mw, H + 34), (20, 20, 20))
        canvas.paste(images[k], (0, 34))
        m = bg.copy()
        d = ImageDraw.Draw(m)
        if len(trail) > 1:
            d.line(trail, fill=(255, 220, 0), width=2)
        x, y = trail[-1]
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 40, 40))
        canvas.paste(m, (images[k].width, 34))
        ok = r.pred_smooth == r.true
        d = ImageDraw.Draw(canvas)
        d.text((6, 3), f"t={r.t:5.1f}s  true: {zones.names[int(r.true)]}", fill=(255, 255, 255))
        d.text((6, 18), f"predicted: {zones.names[int(r.pred_smooth)]} ({r.conf:.2f})",
               fill=(90, 255, 90) if ok else (255, 90, 90))
        out.append(canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.LANCZOS))
    pal = out[len(out) // 2].quantize(colors=64, method=Image.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in out]
    q[0].save(path, save_all=True, append_images=q[1:], duration=200, loop=0, optimize=True)


def recorded(a, cfg, zones, clf):
    split = json.loads((TEST2 / cfg["results_dir"] / "metrics.json").read_text())["split"]
    a.run = a.run or split["test"][0]
    run_dir = TEST2 / cfg["data_dir"] / a.run
    assert a.run in split["test"], f"{a.run} is not a test run (choose one of {split['test']})"
    st = pd.read_csv(run_dir / "states.csv", usecols=["t", "frame", "zone_id", "base_x", "base_y"])
    st = st[(st.frame >= 0) & (st.zone_id >= 0)].reset_index(drop=True)
    images, rows = [], []
    for r in st.itertuples():
        im = Image.open(run_dir / "images" / f"{r.frame:06d}.jpg").convert("RGB")
        p = clf(np.asarray(im))
        rows.append({"t": r.t, "true": r.zone_id, "pred": int(p.argmax()), "conf": float(p.max()),
                     "base_x": r.base_x, "base_y": r.base_y})
        images.append(im)
    return pd.DataFrame(rows), images, f"recorded test run {a.run}"


def live(a, cfg, zones, clf):
    from go2sim.go2_env import CrimeSceneGo2
    from go2sim.static_map import StaticMap
    from go2sim.waypoint_follower import WaypointFollower
    from q3_rl.extract_layouts import scenario_spec
    spec = scenario_spec(a.scenario) if a.scenario[0] in "rs" else a.scenario
    dog = CrimeSceneGo2(spec)
    smap = StaticMap(dog.model, dog.data, dog.sim.robot_geoms)
    rng = np.random.default_rng(a.seed)
    dog.set_lighting(headlight_ambient=0.8, headlight_diffuse=0.9, lamps=(0.3, 0.2, 0.3))
    rows, images = [], []
    state = {"n": 0}
    t0 = dog.time

    def on_tick(d):
        state["n"] += 1
        if state["n"] % 10:
            return
        rgb = d.render_front((320, 240))
        p = clf(rgb)
        x, y, _ = d.pose2d()
        z = zones.zone_id((x, y))
        if z < 0:
            return
        rows.append({"t": d.time - t0, "true": z, "pred": int(p.argmax()), "conf": float(p.max()), "base_x": x, "base_y": y})
        images.append(Image.fromarray(rgb))

    try:
        reach = smap.reachable_from(dog.pose2d()[:2])
        fol = WaypointFollower(dog, smap, on_tick=on_tick)
        for zname in ["desk_entry_area", "lounge_area", "victim_area", "bed_area", "desk_entry_area", "bathroom"]:
            g = smap.sample_free(rng, zones.polygons[zones.names.index(zname)], 0.45, reach)
            if g is None:                      # e.g. r1854: a storage box blocks the bathroom door
                print(f"  {zname}: no reachable free spot in this scene, skipped", flush=True)
                continue
            fol.go_to(g, speed=0.35)
            fol.spin(np.pi)
    finally:
        dog.close()
    return pd.DataFrame(rows), images, f"LIVE in MuJoCo, scene {a.scenario}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--run", help="a test run (default: the first test run of the split)")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--scenario", default="r1854", help="r1854 (unseen by Q1, default), s1/s2 or 1/2 (training scenes)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cfg = load_cfg()
    zones = Zones()
    clf = Classifier(TEST2 / cfg["results_dir"] / a.model / "best.pt", zones)
    df, images, title = live(a, cfg, zones, clf) if a.live else recorded(a, cfg, zones, clf)
    df["pred_smooth"] = majority(df.pred.to_numpy())
    name = f"live_{a.scenario}" if a.live else a.run
    out = TEST2 / cfg["results_dir"]
    acc, accs = timeline_plot(df, zones, title, out / f"demo_{name}.png")
    df.to_csv(out / f"demo_{name}.csv", index=False)
    make_gif(images, df, zones, out / f"demo_{name}.gif")
    print(f"{title}: {len(df)} frames, accuracy {acc:.3f}, smoothed {accs:.3f} -> {out}/demo_{name}.png/.gif")


if __name__ == "__main__":
    main()
