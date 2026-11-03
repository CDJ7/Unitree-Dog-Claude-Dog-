#!/usr/bin/env python3
"""Authoring helper: what is at (x, y)? Prints the top visible surface height and object name.

    python scripts/probe_scene.py --env scenesmith_house_185 --xy 2.4 0.55 --xy 2.8 1.6
    python scripts/probe_scene.py --env scenesmith_house_185 --ray 1.1 2.2 1.55 -1 0 0.1   # where does a ray land?
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401  (first: picks the offscreen GL backend before mujoco loads)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim_env.runtime import SimSession  # noqa: E402
from sim_env.spatter.mj_surfaces import MjSurfaces  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="scenesmith_house_185")
    ap.add_argument("--xy", nargs=2, type=float, action="append", default=[])
    ap.add_argument("--ray", nargs=6, type=float, action="append", default=[], metavar=("X", "Y", "Z", "DX", "DY", "DZ"))
    a = ap.parse_args()
    sim = SimSession(a.env)
    try:
        surf = MjSurfaces(sim.model, sim.data, exclude_geoms=sim.robot_geoms)
        name = lambda g: mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for x, y in a.xy:
            t, g = surf.cast([x, y, 3.0], [0, 0, -3.5])
            print(f"xy=({x:.2f},{y:.2f}): " + (f"top z={3.0 - 3.5 * t:.3f} on {name(g)}" if t is not None else "nothing"))
        for x, y, z, dx, dy, dz in a.ray:
            d = np.array([dx, dy, dz]) / np.linalg.norm([dx, dy, dz])
            t, g = surf.cast([x, y, z], d * 20)
            print(f"ray from ({x},{y},{z}) dir {d.round(2)}: " +
                  (f"hits {name(g)} at {np.round(np.array([x, y, z]) + d * 20 * t, 3)}" if t is not None else "nothing"))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
