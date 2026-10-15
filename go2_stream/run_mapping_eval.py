#!/usr/bin/env python3
"""Scripted survey of the simulated room -> object map -> accuracy report against the sim's true positions.

    python run_mapping_eval.py --gt            # GROUND TRUTH labels (isolates the 3D-position estimate from detection)
    python run_mapping_eval.py --yolo          # real YOLO on the rendered camera (needs OpenGL; best with photo billboards)

Writes results/object_map.json, map_topdown.png, error_per_object.png, accuracy.csv.
The robot visits waypoints and spins 360 degrees at each so every object is seen from several places.
"""
import argparse
import logging
import math
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
log = logging.getLogger("mapping_eval")

WAYPOINTS = [(-2.2, 0.0), (-1.6, -1.8), (0.6, -1.9), (2.5, -0.2), (0.6, 1.9), (-1.8, 0.9)]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def goto(world, client, wp, speed, tol=0.15, timeout=45):
    t0, last_prog, best = time.monotonic(), time.monotonic(), 1e9
    while time.monotonic() - t0 < timeout:
        x, y, yaw = world.pose
        dx, dy = wp[0] - x, wp[1] - y
        dist = math.hypot(dx, dy)
        if dist < tol:
            return True
        err = wrap(math.atan2(dy, dx) - yaw)
        vyaw = max(-0.9, min(0.9, 1.5 * err))
        vx = 0.0 if abs(err) > 0.35 else min(speed, 0.8 * dist + 0.1)
        client.Move(vx, 0.0, vyaw)
        if dist < best - 0.02:
            best, last_prog = dist, time.monotonic()
        elif time.monotonic() - last_prog > 3.0:
            log.warning("stuck on the way to %s, skipping", wp)
            client.StopMove()
            return False
        time.sleep(0.05)
    client.StopMove()
    return False


def spin(world, client, vyaw, turns=1.0):
    total, last = 0.0, world.pose[2]
    t0 = time.monotonic()
    while total < 2 * math.pi * turns and time.monotonic() - t0 < 60:
        client.Move(0.0, 0.0, vyaw)
        time.sleep(0.05)
        yaw = world.pose[2]
        total += abs(wrap(yaw - last))
        last = yaw
    client.StopMove()


def main():
    ap = argparse.ArgumentParser(description="Sim survey + object-map accuracy report")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gt", action="store_true", help="GROUND TRUTH labels from the sim (default)")
    g.add_argument("--yolo", action="store_true", help="YOLO on the rendered camera")
    ap.add_argument("--no-render", action="store_true", help="skip OpenGL; flat-colour ray-cast view (YOLO won't work)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--sim-config", default=os.path.join(HERE, "config", "sim.yaml"))
    ap.add_argument("--det-config", default=os.path.join(HERE, "config", "detection.yaml"))
    ap.add_argument("--map-config", default=os.path.join(HERE, "config", "mapping.yaml"))
    ap.add_argument("--model")
    ap.add_argument("--speed", type=float, default=0.5, help="sim survey speed m/s (the sim has no speed caps)")
    ap.add_argument("--spin-speed", type=float, default=0.8)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=float)
    ap.add_argument("--save-raw", metavar="FILE", help="also pickle every raw sighting (for tuning the filters offline)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from sim.world import SimWorld, SimSportClient, load_config
    from sim.ground_truth import GroundTruthDetector
    from mapping.sim_depth import make_sim_fuse_fn
    from mapping.object_map import ObjectMap, evaluate
    from mapping.observations import usable_observations
    from mapping import report
    from perception.pipeline import DetectionThread

    scfg = load_config(args.sim_config)
    scfg["start_pose"]["standing"] = False
    for k in ("width", "height", "fps"):
        if getattr(args, k):
            scfg["camera"][k] = getattr(args, k)
    dcfg = yaml.safe_load(open(args.det_config))
    mcfg = yaml.safe_load(open(args.map_config))
    world = SimWorld(scfg, render=False if args.no_render else None)
    use_yolo = args.yolo
    if use_yolo:
        from perception.detector import YoloDetector
        m = dcfg["model"]
        detector = YoloDetector(args.model or m["path"], m["conf"], m["device"], m.get("classes") or None, m.get("imgsz", 640))
    else:
        detector = GroundTruthDetector(world)
    labels = "YOLO" if use_yolo else "GROUND TRUTH labels"

    objmap = ObjectMap(mcfg)
    n_obs = [0]
    raw = []

    def on_result(res):
        b = world.bundle_for(res.frame)
        if b is None:
            return
        if args.save_raw:
            raw.append((res.frame.shape, list(res.detections), b.cam_pos.copy(), b.robot_xyyaw[:2]))
        for d in usable_observations(res.detections, res.frame.shape, mcfg["occlusion_overlap_frac"], mcfg["border_margin_px"]):
            if True:
                objmap.add(d.name, d.xyz_world, d.distance_m, ray_origin=b.cam_pos, robot_xy=b.robot_xyyaw[:2])
                n_obs[0] += 1

    world.start()
    thread = DetectionThread(detector, world.buffer, dcfg, fuse_fn=make_sim_fuse_fn(world, dcfg["fusion"]), on_result=on_result)
    thread.start()
    client = SimSportClient(world)
    t_start = time.monotonic()
    try:
        time.sleep(1.0)
        client.StandUp()
        while not world.standing:
            time.sleep(0.05)
        for i, wp in enumerate(WAYPOINTS):
            log.info("waypoint %d/%d %s", i + 1, len(WAYPOINTS), wp)
            goto(world, client, wp, args.speed)
            spin(world, client, args.spin_speed)
        client.StopMove()
        time.sleep(0.5)
    finally:
        thread.stop()
        world.stop()
    if args.save_raw:
        import pickle
        pickle.dump(raw, open(args.save_raw, "wb"))
    truth = [(o.name, o.cls, o.center) for o in world.objects]
    ev = evaluate(objmap, truth, mcfg["match_dist_m"])
    meta = {"labels": labels, "render": world.render_backend, "camera": scfg["camera"], "n_observations": n_obs[0],
            "survey_seconds": round(time.monotonic() - t_start, 1), "center_correction": mcfg["use_center_correction"],
            "note": "sim; estimate = median depth in the central box region, nearest half, deprojected with sim intrinsics"}
    report.write_all(args.out, objmap, world, ev, meta)
    s = ev["summary"]
    print(f"\nLabels: {labels}   render: {world.render_backend}   observations: {n_obs[0]}   survey: {meta['survey_seconds']} s")
    print(f"{'object':22s} {'seen':>5s} {'n_obs':>6s} {'err3D':>7s} {'errXY':>7s} {'surface err':>12s}")
    for r in ev["per_object"]:
        if r["detected"]:
            print(f"{r['object']:22s} {'yes':>5s} {r['n_observations']:6d} {r['error_3d_m']:7.2f} {r['error_xy_m']:7.2f} {r['surface_error_3d_m']:12.2f}")
        else:
            print(f"{r['object']:22s} {'NO':>5s}")
    print(f"detected {s['n_detected']}/{s['n_true']}  mean {s['mean_error_3d_m']} m  median {s['median_error_3d_m']} m  "
          f"max {s['max_error_3d_m']} m  false positives {s['n_false_positives']}")
    print("Wrote", os.path.join(args.out, "object_map.json"), "+ map_topdown.png, error_per_object.png, accuracy.csv")


if __name__ == "__main__":
    main()
