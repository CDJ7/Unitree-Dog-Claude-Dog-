#!/usr/bin/env python3
"""YOLO object detection on the Go2 front camera, with distance/3D position from LiDAR.

Perception only: nothing here moves the robot unless you pass --teleop (then the same keyboard
teleop as main.py runs alongside).

    python run_detection.py --image photo.jpg                  # detector alone, no robot
    python run_detection.py --iface enp0s31f6                  # live camera + LiDAR fusion
    python run_detection.py --iface enp0s31f6 --classes person chair --log det.csv
    python run_detection.py --iface enp0s31f6 --teleop         # also drive with WASD (see README)
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


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    dcfg = yaml.safe_load(open(args.det_config))
    if args.image:
        return run_image(args, dcfg)

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

    camera = lidar_src = state = teleop = det_thread = logger = None
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
        det_thread = DetectionThread(detector, camera.buffer, dcfg, lidar_src, state, lidar_mod, logger)
        det_thread.start()

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
                                            res.lidar_uvr, show_lidar, res.status)
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
        if teleop:
            teleop.shutdown()          # always ends with StopMove
        if det_thread: det_thread.stop()
        if camera: camera.stop()
        if logger: logger.close()
        cv2.destroyAllWindows()
        log.info("Exited cleanly")


if __name__ == "__main__":
    main()
