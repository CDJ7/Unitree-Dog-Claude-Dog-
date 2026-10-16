#!/usr/bin/env python3
"""YOLO object detection on the Go2 front camera, with distance/3D position from LiDAR.

Perception only: nothing here moves the robot unless you pass --teleop (then the same keyboard
teleop as main.py runs alongside).

    python run_detection.py --image photo.jpg                  # detector alone, no robot
    python run_detection.py --iface enp0s31f6                  # live camera + LiDAR fusion
    python run_detection.py --iface enp0s31f6 --classes person chair --log det.csv
    python run_detection.py --iface enp0s31f6 --teleop         # also drive with WASD (see README)
    python run_detection.py --sim                              # MuJoCo sim instead of the robot (teleop on by default)
    python run_detection.py --sim --mission chair              # sim: walk up to a chair and sit (B = go-ahead)
    python run_detection.py --iface enp0s31f6 --teleop --mission chair --mission-dry-run   # robot: print only, no motion
    python run_detection.py --sim --gt                         # sim with GROUND TRUTH labels instead of YOLO
"""
import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("run_detection")


def parse_args():
    ap = argparse.ArgumentParser(description="Go2 YOLO detection with LiDAR distance")
    ap.add_argument("--iface", help="Ethernet interface on 192.168.123.x (default: config.yaml network.iface)")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--det-config", default=os.path.join(HERE, "config", "detection.yaml"))
    ap.add_argument("--model", help="YOLO weights (default from detection.yaml, e.g. yolov8n.pt)")
    ap.add_argument("--conf", type=float, help="confidence threshold")
    ap.add_argument("--classes", nargs="+", help="only these class names, e.g. person chair bottle")
    ap.add_argument("--device", help="auto | cpu | cuda:0")
    ap.add_argument("--no-display", action="store_true", help="headless: no windows (use with --log)")
    ap.add_argument("--log", metavar="FILE", help="write detections to FILE (.csv or .jsonl)")
    ap.add_argument("--no-lidar", action="store_true", help="detection only; distances will be n/a")
    ap.add_argument("--image", metavar="PATH", help="run the detector on one image and exit (no robot, no DDS)")
    ap.add_argument("--teleop", action="store_true", help="also run keyboard teleop (off by default)")
    ap.add_argument("--max-speed", type=float, help="with --teleop: max forward speed m/s")
    ap.add_argument("--sim", action="store_true", help="use the MuJoCo simulation instead of the robot (no DDS)")
    ap.add_argument("--sim-config", default=os.path.join(HERE, "config", "sim.yaml"))
    ap.add_argument("--no-render", action="store_true", help="sim: skip OpenGL, use a flat-colour ray-cast view")
    ap.add_argument("--gt", action="store_true", help="sim: label objects with GROUND TRUTH from the sim, not YOLO")
    ap.add_argument("--no-teleop", action="store_true", help="sim: do not start keyboard teleop (it is on by default)")
    ap.add_argument("--map", action="store_true",
                    help="build an object map from the detections (needs distances: sim depth, or LiDAR on the robot) "
                         "and save results/object_map.json on exit")
    ap.add_argument("--map-config", default=os.path.join(HERE, "config", "mapping.yaml"))
    ap.add_argument("--mission", nargs="*", metavar="CLASS", default=None,
                    help="enable the approach-and-act mission for these classes (none given = all in config/behaviors.yaml). "
                         "Needs teleop. The robot only moves after R (stand) and B (go-ahead) in the video window.")
    ap.add_argument("--allow-autonomy", action="store_true",
                    help="REAL ROBOT: required before a mission may move the dog (read config/behaviors.yaml first)")
    ap.add_argument("--mission-dry-run", action="store_true",
                    help="mission computes and prints the commands it WOULD send, but nothing is sent to the robot")
    ap.add_argument("--mission-min-conf", type=float, metavar="C",
                    help="override the minimum YOLO confidence for mission targets (config: 0.5). The sim's plain-coloured chairs score "
                         "about 0.5; lowering this on the REAL robot makes mis-identified targets more likely")
    ap.add_argument("--behaviors-config", default=os.path.join(HERE, "config", "behaviors.yaml"))
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    return ap.parse_args()


def build_detector(args, dcfg):
    from perception.detector import YoloDetector
    m = dcfg["model"]
    return YoloDetector(args.model or m["path"], args.conf if args.conf is not None else m["conf"],
                        args.device or m["device"], args.classes or m.get("classes") or None, m.get("imgsz", 640))


def run_image(args, dcfg):
    from perception.visualization import draw_detections
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Could not read image: {args.image}")
    det = build_detector(args, dcfg)
    t0 = time.monotonic()
    dets = det.detect(img)
    log.info("%d detections in %.0f ms", len(dets), (time.monotonic() - t0) * 1000)
    for d in dets:
        print(f"{d.name:12s} conf {d.conf:.2f}  bbox {[round(v) for v in d.bbox]}  depth: n/a (no LiDAR in --image mode)")
    out = os.path.splitext(args.image)[0] + "_det.jpg"
    cv2.imwrite(out, draw_detections(img, dets))
    print("Saved", out)


def setup_mission(args, teleop, det_thread, clearance_fn, real):
    """Create and start the approach-and-act mission, or return None when --mission was not given."""
    if args.mission is None:
        return None
    if teleop is None:
        sys.exit("--mission needs keyboard teleop (add --teleop on the robot). Esc/Space/any control key always aborts it.")
    from behavior.mission import Mission
    bcfg = yaml.safe_load(open(args.behaviors_config))
    allow = (not real) or args.allow_autonomy
    if args.mission_min_conf is not None:
        bcfg["safety"]["min_conf"] = args.mission_min_conf
        if real and args.mission_min_conf < 0.5:
            log.warning("mission min_conf lowered to %.2f on the REAL robot: be extra careful", args.mission_min_conf)
    try:
        m = Mission(bcfg, det_thread.latest, teleop, classes=args.mission or None, clearance_fn=clearance_fn,
                    dry_run=args.mission_dry_run, allow_autonomy=allow)
    except ValueError as e:
        sys.exit(f"mission config problem: {e}")
    teleop.on_manual_input = m.manual_input
    m.start()
    log.warning("MISSION enabled for %s%s. Sequence: R (stand) -> wait for 'READY' in the video -> CLICK the video window "
                "(B only works with that window focused) -> B (go-ahead). ANY control key, Space, Esc or V aborts. "
                "Speeds capped at %.2f m/s.",
                "/".join(m.classes), " [DRY RUN]" if args.mission_dry_run else "", bcfg["safety"]["max_vx"])
    if real and not allow:
        log.warning("Autonomy is DISABLED on the real robot: add --allow-autonomy after reading config/behaviors.yaml.")
    return m


def mission_key(mission, key):
    """Window keys: B = go-ahead, C = confirm the action, V = cancel."""
    if mission is None:
        return
    if key in (ord("b"), ord("B")):
        ok, msg = mission.go()
        log.info("go-ahead %s: %s", "ACCEPTED" if ok else "refused", msg)
    elif key in (ord("c"), ord("C")):
        mission.confirm()
    elif key in (ord("v"), ord("V")):
        mission.abort("cancelled by operator")


def save_screenshot(results_dir, img, tag="screenshot"):
    out = os.path.join(results_dir, "screenshots")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, time.strftime(f"{tag}_%Y%m%d_%H%M%S.png"))
    cv2.imwrite(path, img)
    log.info("Saved %s", path)
    return path


def run_sim(args, dcfg):
    """MuJoCo sim backend. Same detector, display and teleop as the real robot; no DDS, no SDK needed."""
    from sim.world import SimWorld, SimSportClient, load_config
    from sim.topdown import TopDownMap
    from sim.ground_truth import GroundTruthDetector
    from mapping.sim_depth import make_sim_fuse_fn
    from perception.pipeline import DetectionThread, DetectionLogger
    from perception.visualization import draw_detections

    import signal
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # clean exit incl. map save
    cfg = yaml.safe_load(open(args.config))
    world = SimWorld(load_config(args.sim_config), render=False if args.no_render else None)
    gt_det = GroundTruthDetector(world)
    yolo = None if args.gt else build_detector(args, dcfg)
    detector = gt_det if args.gt else yolo
    teleop = det_thread = logger = objmap = mcfg = mission = None
    try:
        world.start()
        if not args.no_teleop:
            from teleop import Teleop
            teleop = Teleop(cfg["control"], SimSportClient(world), max_speed=args.max_speed)
            teleop.start()
            log.info("Dog starts lying down: press R to stand, then WASD. P = screenshot, G = toggle YOLO / GROUND TRUTH.")
        logger = DetectionLogger(args.log) if args.log else None
        objmap = on_result = None
        if args.map:
            from mapping.object_map import ObjectMap
            from mapping.observations import usable_observations
            mcfg = yaml.safe_load(open(args.map_config))
            objmap = ObjectMap(mcfg)

            def on_result(res):
                b = world.bundle_for(res.frame)
                if b is None:
                    return
                for d in usable_observations(res.detections, res.frame.shape, mcfg["occlusion_overlap_frac"], mcfg["border_margin_px"]):
                    objmap.add(d.name, d.xyz_world, d.distance_m, ray_origin=b.cam_pos, robot_xy=b.robot_xyyaw[:2])
        det_thread = DetectionThread(detector, world.buffer, dcfg, logger=logger,
                                     fuse_fn=make_sim_fuse_fn(world, dcfg["fusion"]), on_result=on_result)
        det_thread.start()
        if args.mission is not None:
            from behavior.clearance import make_sim_clearance
            bcfg = yaml.safe_load(open(args.behaviors_config))
            mission = setup_mission(args, teleop, det_thread, make_sim_clearance(world, bcfg["safety"]["corridor"]), real=False)
        topdown = TopDownMap(world)
        win = "go2_sim_detection"
        if not args.no_display:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        last_seq, shown, times = 0, None, []
        while True:
            res = det_thread.latest()
            if res is not None and res.seq != last_seq:
                last_seq = res.seq
                now = time.monotonic()
                times.append(now)
                while times and now - times[0] > 2.0:
                    times.pop(0)
                dfps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
                shown = draw_detections(res.frame, res.detections, res.infer_fps, dfps, None, False,
                                        f"sim ({world.render_backend})  {res.status}",
                                        mission.status_text() if mission else "")
            if args.no_display:
                time.sleep(0.05)
            else:
                if shown is not None:
                    cv2.imshow(win, shown)
                cv2.imshow("go2_sim_map", topdown.draw(objmap.estimates() if objmap else None))
                key = cv2.waitKey(15) & 0xFF
                if key == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                mission_key(mission, key)
                if key in (ord("p"), ord("P")) and shown is not None:
                    save_screenshot(args.results_dir, shown, "detection")
                    save_screenshot(args.results_dir, topdown.draw(), "map")
                if key in (ord("g"), ord("G")):
                    if yolo is None and not args.gt:
                        continue
                    if yolo is None:
                        yolo = build_detector(args, dcfg)      # --gt run: load YOLO on demand
                    det_thread.detector = yolo if det_thread.detector is gt_det else gt_det
                    log.info("Labels now: %s", "GROUND TRUTH" if det_thread.detector is gt_det else "YOLO")
            if teleop and teleop.quit_event.is_set():
                break
    except KeyboardInterrupt:
        log.info("Interrupted")
    except Exception:
        log.exception("Fatal error")
    finally:
        if mission: mission.stop()           # first: drop any autonomous velocity, then stop the robot
        if teleop:
            teleop.shutdown()
        if det_thread: det_thread.stop()
        world.stop()
        if logger: logger.close()
        if args.map and objmap is not None:
            from mapping.object_map import evaluate
            from mapping import report
            truth = [(o.name, o.cls, o.center) for o in world.objects]
            ev = evaluate(objmap, truth, mcfg["match_dist_m"])
            report.write_all(args.results_dir, objmap, world, ev,
                             {"labels": "GROUND TRUTH labels" if det_thread and det_thread.detector is gt_det else "YOLO",
                              "render": world.render_backend, "camera": world.cfg["camera"], "note": "interactive run"})
            log.info("Object map saved to %s (detected %d/%d)", args.results_dir, ev["summary"]["n_detected"], ev["summary"]["n_true"])
        cv2.destroyAllWindows()
        log.info("Exited cleanly")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    dcfg = yaml.safe_load(open(args.det_config))
    if args.image:
        return run_image(args, dcfg)
    if args.sim:
        return run_sim(args, dcfg)

    cfg = yaml.safe_load(open(args.config))
    iface = args.iface or cfg["network"]["iface"]
    if not os.path.isdir(f"/sys/class/net/{iface}"):
        sys.exit(f"Interface '{iface}' not found. Available: {', '.join(os.listdir('/sys/class/net'))}")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from camera import CameraThread
    from perception.pipeline import DetectionThread, DetectionLogger
    from perception.visualization import draw_detections

    log.info("Initialising DDS on %s", iface)
    ChannelFactoryInitialize(0, iface)

    camera = lidar_src = state = teleop = det_thread = logger = objmap_real = mission = None
    try:
        detector = build_detector(args, dcfg)          # load the model first: it can take a few seconds
        camera = CameraThread(cfg["camera"], iface)
        camera.start()
        lidar_mod = None
        if not args.no_lidar:
            import lidar as lidar_mod
            try:
                lidar_src = lidar_mod.LidarSource(cfg["lidar"])
                state = lidar_mod.StateSource(cfg["state"])
            except Exception as e:
                log.error("LiDAR/state subscription failed (%s). Continuing with distances n/a.", e)
                lidar_src = state = lidar_mod = None
        if args.teleop:
            from teleop import Teleop
            teleop = Teleop(cfg["control"], max_speed=args.max_speed)
            teleop.start()
        logger = DetectionLogger(args.log) if args.log else None
        on_result = None
        if args.map and state is not None:
            from mapping.object_map import ObjectMap
            from mapping.real_adapter import make_on_result
            objmap_real = ObjectMap(yaml.safe_load(open(args.map_config)))
            on_result = make_on_result(state, objmap_real)
        det_thread = DetectionThread(detector, camera.buffer, dcfg, lidar_src, state, lidar_mod, logger, on_result=on_result)
        det_thread.start()
        if args.mission is not None:
            if lidar_src is None:
                sys.exit("--mission needs LiDAR distances (drop --no-lidar and fix the LiDAR subscription first).")
            from behavior.clearance import make_real_clearance
            bcfg = yaml.safe_load(open(args.behaviors_config))
            clear = make_real_clearance(lidar_src, state, lidar_mod, dcfg["fusion"].get("lidar_frame", "auto"),
                                        bcfg["safety"]["corridor"])
            mission = setup_mission(args, teleop, det_thread, clear, real=True)

        win = "go2_detection"
        if not args.no_display:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        show_lidar, last_seq, shown, times = False, 0, None, []
        last_stale = time.monotonic()
        while True:
            res = det_thread.latest()
            if res is not None and res.seq != last_seq:
                last_seq = res.seq
                now = time.monotonic()
                times.append(now)
                while times and now - times[0] > 2.0:
                    times.pop(0)
                if not args.no_display:
                    dfps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
                    shown = draw_detections(res.frame, res.detections, res.infer_fps, dfps,
                                            res.lidar_uvr, show_lidar, res.status,
                                            mission.status_text() if mission else "")
            if args.no_display:
                time.sleep(0.05)
            else:
                if shown is not None:
                    cv2.imshow(win, shown)
                key = cv2.waitKey(10) & 0xFF
                if key == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("l"), ord("L")):
                    show_lidar = not show_lidar
                mission_key(mission, key)
            if teleop and teleop.quit_event.is_set():
                break
            if time.monotonic() - last_stale > 1.0:
                last_stale = time.monotonic()
                if lidar_src: lidar_src.check_stale()
                if state: state.check_stale()
    except KeyboardInterrupt:
        log.info("Interrupted")
    except Exception:
        log.exception("Fatal error")
    finally:
        if mission: mission.stop()     # first: drop any autonomous velocity, then stop the robot
        if teleop:
            teleop.shutdown()          # always ends with StopMove
        if det_thread: det_thread.stop()
        if camera: camera.stop()
        if logger: logger.close()
        if args.map and objmap_real is not None:
            import json
            os.makedirs(args.results_dir, exist_ok=True)
            with open(os.path.join(args.results_dir, "object_map_real.json"), "w") as f:
                json.dump({"note": "real robot; LiDAR-based positions with PLACEHOLDER camera-LiDAR extrinsics",
                           "objects": objmap_real.to_dict()}, f, indent=2)
            log.info("Object map saved to %s", args.results_dir)
        cv2.destroyAllWindows()
        log.info("Exited cleanly")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    sys.stdout.flush()
    sys.stderr.flush()
    # main() has already stopped the robot and closed everything. Skip interpreter teardown: live DDS/Open3D threads
    # can abort it ("terminate called without an active exception") after a clean exit.
    os._exit(code)
