#!/usr/bin/env python3
"""A1 acceptance check, headless: the Go2 loads, stands, walks on cmd_vel and every sensor works.

    python scripts/check_sim.py                       # houses 185 and 186
    python scripts/check_sim.py --env scenesmith_house_185 --json docs/media/a1_check.json

For each environment: stand still, then forward, backward, turn left/right and
strafe, each checked against the command (direction and rough distance);
RGB not blank; depth valid and consistent with an independent ray cast;
LiDAR scan and cloud populated; IMU gravity ~9.8 m/s^2; no collisions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401  (first: picks the offscreen GL backend before mujoco loads)

from sim_env.runtime import SimSession  # noqa: E402  (first: picks the headless GL backend)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

MOVES = [  # name, (vx, vy, wz), seconds, check(dx_body, dy_body, dyaw)
    ("forward", (0.3, 0, 0), 2.0, lambda dx, dy, dyaw: dx > 0.25),
    ("backward", (-0.2, 0, 0), 2.0, lambda dx, dy, dyaw: dx < -0.15),
    ("turn left", (0, 0, 0.5), 2.0, lambda dx, dy, dyaw: dyaw > 0.6),
    ("turn right", (0, 0, -0.5), 2.0, lambda dx, dy, dyaw: dyaw < -0.6),
    ("strafe left", (0, 0.15, 0), 2.0, lambda dx, dy, dyaw: dy > 0.1),
    ("strafe right", (0, -0.15, 0), 2.0, lambda dx, dy, dyaw: dy < -0.1),
    ("stop", (0, 0, 0), 1.0, lambda dx, dy, dyaw: abs(dx) < 0.05 and abs(dy) < 0.05 and abs(dyaw) < 0.1),
]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def check_env(env):
    res = {"environment": env, "checks": []}

    def check(name, ok, detail):
        res["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    t0 = time.time()
    sim = SimSession(env)
    try:
        res["load_s"] = round(time.time() - t0, 1)
        sim.settle(2.0)
        st = sim.state()
        check("stands", st.upright > 0.95 and 0.2 < st.position[2] < 0.4 and np.linalg.norm(st.lin_vel_world) < 0.05,
              f"base z={st.position[2]:.3f} m, upright={st.upright:.3f}")
        collisions = set()
        wall = time.time()
        sim_t = 0.0
        for name, cmd, dur, ok_fn in MOVES:
            if not any(cmd):
                sim.step(0.5)          # let it brake before measuring that it stays put
            s0 = sim.state()
            t_end = sim.time + dur
            while sim.time < t_end:
                if any(cmd):
                    sim.cmd_vel(*cmd)
                sim.step(0.05)
                collisions |= {(c.robot_geom, c.other_body) for c in sim.contacts()}
            sim_t += dur
            s1 = sim.state()
            d = s1.position[:2] - s0.position[:2]
            c, s = np.cos(s0.yaw), np.sin(s0.yaw)
            dx, dy = c * d[0] + s * d[1], -s * d[0] + c * d[1]
            dyaw = wrap(s1.yaw - s0.yaw)
            check(f"walk {name}", ok_fn(dx, dy, dyaw) and s1.upright > 0.9,
                  f"cmd {cmd} for {dur}s -> dx={dx:+.2f} m, dy={dy:+.2f} m, dyaw={np.degrees(dyaw):+.0f} deg")
        res["realtime_factor"] = round(sim_t / (time.time() - wall), 2)

        f = sim.camera()
        check("RGB camera", f.rgb.shape == (240, 320, 3) and f.rgb.std() > 5,
              f"{f.rgb.shape[1]}x{f.rgb.shape[0]}, pixel std {f.rgb.std():.1f}")
        valid = np.isfinite(f.depth_m)
        # independent ray cast through 9 pixels vs the depth image
        errs = []
        for u in (40, 160, 280):
            for v in (40, 120, 200):
                ray_cam = np.array([(u - f.K[0, 2]) / f.K[0, 0], (v - f.K[1, 2]) / f.K[1, 1], 1.0])
                R, p = f.T_world_cam[:3, :3], f.T_world_cam[:3, 3]
                dw = R @ ray_cam
                n = np.linalg.norm(dw)
                gid = np.zeros(1, np.int32)
                dist = mujoco.mj_ray(sim.model, sim.data, p, dw / n, np.array([1, 0, 0, 1, 0, 1], np.uint8),
                                     1, int(sim.model.cam_bodyid[sim.camera_mj_id]), gid)
                if dist > 0 and np.isfinite(f.depth_m[v, u]):
                    errs.append(abs(dist / n - f.depth_m[v, u]))
        check("depth", valid.mean() > 0.5 and len(errs) >= 5 and np.median(errs) < 0.05,
              f"{valid.mean() * 100:.0f}% valid, range {np.min(f.depth_m[valid]):.2f}-{np.max(f.depth_m[valid]):.2f} m, "
              f"median error vs ray cast {np.median(errs) * 1000:.1f} mm (depth is ray cast at 160x120, upsampled)")
        scan = sim.lidar()
        n_ok = int(np.isfinite(scan.ranges).sum())
        check("LiDAR 2D scan", n_ok > 300, f"{n_ok}/{scan.ranges.size} returns, nearest {np.min(scan.ranges):.2f} m")
        check("LiDAR 3D cloud", len(scan.points_lidar) > 500, f"{len(scan.points_lidar)} points")
        imu = sim.state().imu
        g = np.linalg.norm(imu.accel)
        check("IMU", 8.5 < g < 11.0 and np.linalg.norm(imu.gyro) < 0.5,
              f"|accel|={g:.2f} m/s2 standing, |gyro|={np.linalg.norm(imu.gyro):.3f} rad/s")
        st = sim.state()
        check("odometry/pose", np.all(np.isfinite(st.position)) and abs(np.linalg.norm(st.quat_wxyz) - 1) < 1e-6,
              f"x={st.position[0]:.2f} y={st.position[1]:.2f} yaw={np.degrees(st.yaw):.1f} deg")
        check("no collisions", not collisions, f"{sorted(collisions) or 'none'}")
        res["camera_K"] = f.K.round(3).tolist()
    finally:
        sim.close()
    res["ok"] = all(c["ok"] for c in res["checks"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", action="append", help="environment id (repeatable)")
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args()
    envs = args.env or ["scenesmith_house_185", "scenesmith_house_186"]
    results = []
    for env in envs:
        print(f"== {env}")
        results.append(check_env(env))
        print(f"   load {results[-1].get('load_s')} s, sim runs at {results[-1].get('realtime_factor')}x real time")
    ok = all(r["ok"] for r in results)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"ok": ok, "results": results}, indent=1))
    print("A1 CHECK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
