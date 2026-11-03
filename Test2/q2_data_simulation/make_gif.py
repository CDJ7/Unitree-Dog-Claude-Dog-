#!/usr/bin/env python3
"""Q2: GIF of the Go2 walking through the crime scene (chase camera | head camera | map).

    python q2_data_simulation/make_gif.py                    # -> results/q2/go2_crime_scene.gif

The dog runs a short tour of all five zones in scenario S1 with the learned locomotion policy and the
waypoint follower; every 0.25 s of sim time one GIF frame is rendered (sim time is shown, so the
GIF plays at ~2x real time at 8 fps).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

from go2sim import paths  # noqa: E402,F401
from PIL import Image, ImageDraw  # noqa: E402

from go2sim.go2_env import CrimeSceneGo2  # noqa: E402
from go2sim.static_map import StaticMap  # noqa: E402
from go2sim.waypoint_follower import WaypointFollower  # noqa: E402
from go2sim.zones import Zones  # noqa: E402

EXTENT = (-0.5, 8.7, -0.5, 5.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="1")
    ap.add_argument("--out", default=str(TEST2 / "results/q2/go2_crime_scene.gif"))
    ap.add_argument("--every", type=float, default=0.25, help="sim seconds per GIF frame")
    ap.add_argument("--fps", type=float, default=8)
    a = ap.parse_args()
    zones = Zones()
    dog = CrimeSceneGo2(a.scenario)
    smap = StaticMap(dog.model, dog.data, dog.sim.robot_geoms)
    bg_path = TEST2 / "results/q2/house185_topdown.png"
    bg = Image.open(bg_path).convert("RGB") if bg_path.exists() else Image.new("RGB", (920, 550), "gray")
    W, H = 320, 240
    mh = H
    mw = int(round(bg.width * mh / bg.height))
    bg = bg.resize((mw, mh))
    frames, trail = [], []
    n = int(round(a.every / 0.02))
    state = {"k": 0}

    def to_px(x, y):
        return ((x - EXTENT[0]) / (EXTENT[1] - EXTENT[0]) * mw, (EXTENT[3] - y) / (EXTENT[3] - EXTENT[2]) * mh)

    def on_tick(d):
        state["k"] += 1
        x, y, yaw = d.pose2d()
        if state["k"] % 5 == 0:
            trail.append(to_px(x, y))
        if state["k"] % n:
            return
        chase = Image.fromarray(d.render_chase((W, H), distance=1.8, elevation=-22))
        front = Image.fromarray(d.render_front((W, H)))
        m = bg.copy()
        dr = ImageDraw.Draw(m)
        if len(trail) > 1:
            dr.line(trail, fill=(255, 220, 0), width=2)
        px, py = to_px(x, y)
        dr.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 40, 40))
        dr.line([px, py, px + 12 * np.cos(yaw), py - 12 * np.sin(yaw)], fill=(255, 40, 40), width=3)
        canvas = Image.new("RGB", (2 * W + mw, H + 22), (20, 20, 20))
        canvas.paste(chase, (0, 22))
        canvas.paste(front, (W, 22))
        canvas.paste(m, (2 * W, 22))
        c = d.command
        ImageDraw.Draw(canvas).text(
            (6, 5), f"t={d.time - t0:5.1f}s  zone: {zones.zone_name((x, y)):16s} cmd vx={c[0]:+.2f} vy={c[1]:+.2f} "
                    f"wz={c[2]:+.2f}   [chase | Go2 head camera | map]", fill=(255, 255, 255))
        frames.append(canvas)

    try:
        dog.teleport(5.0, 1.2, np.pi * 0.75)
        t0 = dog.time
        fol = WaypointFollower(dog, smap, on_tick=on_tick)
        tour = [(3.4, 2.15, None), (1.35, 2.2, np.pi), (2.1, 3.4, None), (4.7, 2.2, None),
                (6.6, 2.1, np.pi * 1.2), (7.3, 2.6, None), (5.0, 1.0, None)]
        for x, y, spin in tour:
            fol.go_to((x, y), speed=0.4)
            if spin:
                fol.spin(spin)
    finally:
        dog.close()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / a.fps), loop=0, optimize=True)
    print(f"{len(frames)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
