#!/usr/bin/env python3
"""First check on the real Go2: sensors only, the dog does NOT move (no SportClient is created).

    python scripts/real_check.py                 # Ethernet, interface auto-detected
    python scripts/real_check.py --seconds 20    # longer: walk the dog around with the remote meanwhile
    python scripts/real_check.py --link sim      # the same check against the simulator

It reads the odometry pose, the LiDAR (rt/utlidar/cloud_deskewed), and the camera for --seconds, builds an
occupancy map from them, and classifies a few camera frames with the lab zone classifier. Then it prints
PASS / WARN / FAIL lines and writes outputs/real_check/<time>/: map.png (occupancy + scan), frame.jpg,
report.json. Run it before map_zones.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--link", choices=["go2", "sim"], default="go2")
    ap.add_argument("--iface")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--body-height", type=float, default=0.31, help="floor distance below the body origin, m")
    ap.add_argument("--model", help="zone classifier (default: the lab model from Test 2)")
    a = ap.parse_args()
    from navigation.occupancy import OccupancyGrid
    from real_robot.session import Go2Link, RealGo2Session, SimLink
    if a.link == "sim":
        from sim_env.launch import session_from_args
        from sim_env.scenarios.random_sim import write_random
        ns = argparse.Namespace(scenario=str(write_random(1854, env="scenesmith_house_185")), random=None, env=None,
                                spawn=None, no_stains=False, rebuild=False)
        link = SimLink(session_from_args(ns)[0])
    else:
        link = Go2Link(iface=a.iface, control=False)
    s = RealGo2Session(link, keys=False, body_height=a.body_height)
    out = ROOT / "outputs" / "real_check" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    st = s.state()
    grid = OccupancyGrid(origin=st.position[:2] - np.array([8.0, 8.0]), size=(16.0, 16.0))
    n_scan, n_pts, poses, rel_z = 0, [], [], []
    t0 = s.time
    while s.time - t0 < a.seconds:
        scan = s.lidar()
        grid.update(scan)
        n_scan += 1
        n_pts.append(int(np.isfinite(scan.ranges).sum()))
        x, y, z, yaw, _ = link.pose()
        poses.append((x, y, yaw))
        raw = np.vstack([c[0] for c in s._clouds]) if s._clouds else np.zeros((0, 3))
        d = np.hypot(raw[:, 0] - x, raw[:, 1] - y)
        near = raw[(d > 0.6) & (d < 4.0)]
        if len(near) > 30:
            rel_z.append(float(np.percentile(near[:, 2], 2) - z))        # floor relative to the body
        link.advance(0.2) if a.link == "go2" else s.step(0.2)
    checks = []

    def check(name, ok, detail, warn=False):
        checks.append({"check": name, "result": "PASS" if ok else ("WARN" if warn else "FAIL"), "detail": detail})

    check("odometry pose", len(poses) > 0, f"{len(poses)} poses, last {np.round(poses[-1], 2).tolist() if poses else None}")
    frac = float(np.mean([p > 0 for p in n_pts])) if n_pts else 0.0
    check("LiDAR scans", frac > 0.8, f"{n_scan} scans, median {int(np.median(n_pts)) if n_pts else 0}/360 directions with a return")
    floor = float(np.median(rel_z)) if rel_z else None
    if a.link == "sim":
        check("floor height", True, "skipped: the simulated LiDAR returns no floor points (real dog only)")
    else:
        # the session puts the floor body_height below the body; the scan band starts 0.12 m above it
        check("floor height", floor is not None and abs(floor + s.body_height) < 0.06,
              f"floor {-floor:.2f} m below the body (the session assumes {s.body_height:.2f} m; if they differ by "
              "more than 0.06 m, pass --body-height)" if floor is not None else "no floor points near the dog",
              warn=True)
    free_m2 = float(grid.free().sum()) * grid.res ** 2
    check("occupancy map", free_m2 > 2.0, f"{free_m2:.1f} m2 free, {int(grid.occupied().sum())} occupied cells")
    cam = s.camera()
    check("camera", cam is not None, f"{cam.rgb.shape[1]}x{cam.rgb.shape[0]}" if cam is not None else "no frame")
    zone = None
    if cam is not None:
        import cv2
        cv2.imwrite(str(out / "frame.jpg"), cv2.cvtColor(cam.rgb, cv2.COLOR_RGB2BGR))
        from perception.zone_classifier import REAL_MODEL, SIM_MODEL, ZoneClassifier
        model = Path(a.model or (SIM_MODEL if a.link == "sim" else REAL_MODEL))
        if model.exists():
            clf = ZoneClassifier(model)
            p = clf(cam.rgb)
            zone = (clf.classes[int(np.argmax(p))], round(float(p.max()), 3))
            check("zone classifier", True, f"{model.name}: camera sees {zone[0]} ({zone[1]})")
        else:
            check("zone classifier", False, f"{model} not found (copy best.pt there; it is not in git)")
    from mapping.zone_map import ZoneMap
    import cv2
    img = ZoneMap(grid, ["-"]).image(np.array(poses)[:, :2], robot=s.state().pose2d)
    cv2.imwrite(str(out / "map.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    for c in checks:
        print(f"{c['result']:5s} {c['check']:16s} {c['detail']}")
    (out / "report.json").write_text(json.dumps({"checks": checks, "zone": zone}, indent=1))
    print(f"-> {out}")
    s.close()
    sys.exit(1 if any(c["result"] == "FAIL" for c in checks) else 0)


if __name__ == "__main__":
    main()
