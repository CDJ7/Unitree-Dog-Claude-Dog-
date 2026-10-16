#!/usr/bin/env python3
"""Drive the simulated Go2 with the same keyboard teleop as the real robot (no robot, no DDS needed).

    python run_sim.py                  # camera view + top-down map, WASD drives
    python run_sim.py --no-render      # headless-safe ray-cast view (no OpenGL needed)

Keys are the same as teleop.py: R stand up, F lie down, W/S/A/D/Q/E move, +/- speed, Space stop, Esc quit.
For YOLO detection in the sim use run_detection.py --sim.
"""
import argparse
import logging
import os
import sys
import time

import cv2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
log = logging.getLogger("run_sim")


def main():
    ap = argparse.ArgumentParser(description="Go2 MuJoCo sim with keyboard teleop")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--sim-config", default=os.path.join(HERE, "config", "sim.yaml"))
    ap.add_argument("--max-speed", type=float)
    ap.add_argument("--no-render", action="store_true", help="skip OpenGL; flat-colour ray-cast view instead")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    from sim.world import SimWorld, SimSportClient, load_config
    from sim.topdown import TopDownMap
    from teleop import Teleop
    from camera import draw_overlay

    cfg = yaml.safe_load(open(args.config))
    world = SimWorld(load_config(args.sim_config), render=False if args.no_render else None)
    world.start()
    teleop = Teleop(cfg["control"], SimSportClient(world), max_speed=args.max_speed)
    topdown = TopDownMap(world)
    try:
        teleop.start()
        log.info("Dog starts lying down. Press R to stand, then drive. Esc quits.")
        last_seq = 0
        while not teleop.quit_event.is_set():
            frame, seq, _ = world.buffer.get()
            if not args.no_display:
                if frame is not None:
                    snap = teleop.snapshot()
                    cv2.imshow("sim_camera", draw_overlay(frame, world.fps, (snap["vx"], snap["vy"], snap["vyaw"]),
                                                          snap["scale"], snap["status"], [f"render: {world.render_backend}"]))
                cv2.imshow("sim_map", topdown.draw())
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.shutdown()
        world.stop()
        cv2.destroyAllWindows()
        log.info("Exited cleanly")


if __name__ == "__main__":
    main()
