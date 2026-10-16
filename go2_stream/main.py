#!/usr/bin/env python3
"""Go2 camera + LiDAR streaming with keyboard teleop. See README.md.

    python main.py --iface enp0s31f6
    python main.py --iface enp0s31f6 --no-control          # view only
"""
import argparse
import logging
import os
import signal
import sys
import time

import cv2
import numpy as np
import yaml

log = logging.getLogger("main")


def parse_args():
    ap = argparse.ArgumentParser(description="Go2 camera + LiDAR viewer with keyboard teleop")
    ap.add_argument("--iface", help="Ethernet interface on 192.168.123.x (default: config.yaml network.iface)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-control", action="store_true", help="view-only: never send motion commands")
    ap.add_argument("--max-speed", type=float, metavar="M_S",
                    help="max forward speed in m/s (vy/vyaw scale proportionally); capped by config hard caps")
    ap.add_argument("--camera-mode", choices=["videoclient", "multicast"])
    ap.add_argument("--viewer", choices=["open3d", "topdown"])
    return ap.parse_args()


def check_iface(iface):
    if not os.path.isdir(f"/sys/class/net/{iface}"):
        log.error("Interface '%s' not found. Available: %s. Run ./setup_network.sh detect",
                  iface, ", ".join(os.listdir("/sys/class/net")))
        sys.exit(2)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    iface = args.iface or cfg["network"]["iface"]
    if args.camera_mode:
        cfg["camera"]["mode"] = args.camera_mode
    if args.viewer:
        cfg["lidar"]["viewer"] = args.viewer
    check_iface(iface)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    log.info("Initialising DDS on %s", iface)
    ChannelFactoryInitialize(0, iface)

    camera = lidar = state = viewer = teleop = None
    try:
        if not args.no_camera:
            from camera import CameraThread, draw_overlay
            camera = CameraThread(cfg["camera"], iface)
            camera.start()
        if not args.no_lidar:
            import lidar as lidar_mod
            lidar = lidar_mod.LidarSource(cfg["lidar"])
        state = None
        try:
            import lidar as lidar_mod
            state = lidar_mod.StateSource(cfg["state"])
        except Exception as e:
            log.warning("State subscription unavailable: %s", e)
        if not args.no_control:
            from teleop import Teleop
            teleop = Teleop(cfg["control"], max_speed=args.max_speed)
            teleop.start()
        else:
            log.info("View-only mode: no motion commands will be sent")
        if lidar is not None:
            viewer = lidar_mod.make_viewer(cfg["lidar"])

        cam_win = cfg["camera"]["window"]
        if camera is not None:
            cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)
            placeholder = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(placeholder, "waiting for camera...", (150, 190), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (200, 200, 200), 2)
        last_cam_seq = last_lidar_seq = 0
        last_view = 0.0
        view_period = 1.0 / cfg["lidar"]["view_hz"]
        last_stale_check = time.monotonic()
        shown_frame = None

        while True:
            now = time.monotonic()
            snap = teleop.snapshot() if teleop else {"vx": 0, "vy": 0, "vyaw": 0, "scale": 0, "status": "view-only"}

            if camera is not None:
                frame, seq, _ = camera.buffer.get()
                if frame is not None and seq != last_cam_seq:
                    last_cam_seq = seq
                    shown_frame = frame
                if shown_frame is not None or seq == 0:
                    base = shown_frame if shown_frame is not None else placeholder
                    extra = state.lines() if state else ()
                    cv2.imshow(cam_win, draw_overlay(base, camera.fps, (snap["vx"], snap["vy"], snap["vyaw"]),
                                                     snap["scale"], snap["status"], extra))

            if lidar is not None and viewer is not None and now - last_view >= view_period:
                msg, seq, _ = lidar.latest.get()
                if msg is not None and seq != last_lidar_seq:
                    last_lidar_seq, last_view = seq, now
                    try:
                        pts = lidar_mod.prepare_points(lidar_mod.pointcloud2_to_xyz(msg), cfg["lidar"])
                        viewer.update(pts, (state.lines() if state else ()))
                    except Exception as e:
                        log.warning("Could not parse/display point cloud: %s", e)
            if viewer is not None and viewer.pump() is False:
                log.info("LiDAR window closed")
                break

            if now - last_stale_check > 1.0:
                last_stale_check = now
                if lidar: lidar.check_stale()
                if state: state.check_stale()

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            if camera is not None and cv2.getWindowProperty(cam_win, cv2.WND_PROP_VISIBLE) < 1:
                log.info("Camera window closed")
                break
            if teleop and teleop.quit_event.is_set():
                log.info("ESC pressed")
                break
    except KeyboardInterrupt:
        log.info("Interrupted")
    except Exception:
        log.exception("Fatal error")
    finally:
        if teleop:
            teleop.shutdown()      # always ends with StopMove
        if camera:
            camera.stop()
        if viewer:
            try: viewer.close()
            except Exception: pass
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
