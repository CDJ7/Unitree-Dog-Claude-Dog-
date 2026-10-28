#!/usr/bin/env python3
"""Drive the simulated Go2 with the keyboard.

    python scripts/teleop.py                 # MuJoCo viewer window (WSLg / desktop)
    python scripts/teleop.py --cv            # OpenCV window: chase view + front camera
    python scripts/teleop.py --scenario 2 --env scenesmith_house_186

Keys (click the window first): W/S forward/back, A/D turn, Q/E strafe,
Space stop, R reset, Esc quit (viewer: close the window). Speeds latch: each
press adds a step, Space zeroes them. Collisions are printed as they happen.
"""
import argparse
import queue
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from navigation.teleop import HELP, KeyTeleop  # noqa: E402
from sim_env.launch import add_sim_args, session_from_args  # noqa: E402


class CollisionLog:
    def __init__(self):
        self.seen = set()
        self.count = 0

    def update(self, sim):
        now = set()
        for c in sim.contacts():
            pair = (c.robot_geom, c.other_body)
            if pair in now:
                continue
            now.add(pair)
            if pair not in self.seen:
                self.count += 1
                print(f"\n[collision] t={c.t:.2f}s robot {c.robot_geom} <-> {c.other_body}/{c.other_geom} "
                      f"at {np.round(c.position, 2).tolist()}", flush=True)
        self.seen = now


def status_line(sim, tele, log):
    st = sim.state()
    x, y, yaw = st.pose2d
    return (f"\r{tele.status()} | pose x={x:+.2f} y={y:+.2f} yaw={np.degrees(yaw):+6.1f} deg "
            f"| upright {st.upright:.2f} | collisions {log.count} | t={st.t:6.1f}s ")


def step_realtime(sim, tele, clock):
    """Advance physics to match wall-clock time, at most 0.25 s per call."""
    if tele.reset_requested:
        sim.reset()
        tele.reset_requested = False
    sim.cmd_vel(*tele.target)
    target = clock["sim0"] + (time.monotonic() - clock["wall0"])
    n = 0
    while sim.time < target and n < 125:
        sim.step(steps=1)
        n += 1
    if sim.time < target - 0.25:   # can't keep up: let the sim run slower than real time
        clock["sim0"] = sim.time
        clock["wall0"] = time.monotonic()


def run_viewer(sim, tele, log):
    import mujoco
    import mujoco.viewer

    try:   # free W/S/A/D/Q/E from MuJoCo's own render shortcuts (simulator's helper)
        from sim.native.runtime import disable_mujoco_viewer_shortcuts
        disable_mujoco_viewer_shortcuts()
    except Exception as e:  # noqa: BLE001
        print(f"(could not disable viewer shortcuts: {e})")
    keys = queue.SimpleQueue()
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=keys.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[3] = 0
            viewer.opt.geomgroup[4] = 0
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = sim.base_body
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 2.2, -25, 180
        clock = {"sim0": sim.time, "wall0": time.monotonic()}
        next_sync = next_print = 0.0
        while viewer.is_running() and not tele.quit_requested:
            while not keys.empty():
                tele.press(keys.get_nowait())
            step_realtime(sim, tele, clock)
            now = time.monotonic()
            if now >= next_sync:
                log.update(sim)
                viewer.sync()
                next_sync = now + 1 / 60
            if now >= next_print:
                print(status_line(sim, tele, log), end="", flush=True)
                next_print = now + 0.25
            time.sleep(0.001)


def run_cv(sim, tele, log):
    import cv2

    clock = {"sim0": sim.time, "wall0": time.monotonic()}
    win = "Go2 teleop (click here, then use the keys)"
    cv2.namedWindow(win)
    next_cam = next_print = next_chase = 0.0
    front = chase = None
    while not tele.quit_requested:
        step_realtime(sim, tele, clock)
        log.update(sim)
        if sim.time >= next_cam:
            front = sim.camera().rgb
            next_cam = sim.time + 0.2
        f = cv2.resize(front, (480, 360), interpolation=cv2.INTER_NEAREST)
        if time.monotonic() >= next_chase:       # third-person view at <= 10 Hz
            chase = sim.render_view(480, 360)
            next_chase = time.monotonic() + 0.1
        img = cv2.cvtColor(np.hstack([chase, f]), cv2.COLOR_RGB2BGR)
        cv2.putText(img, status_line(sim, tele, log).strip()[:110], (8, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, HELP, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, img)
        if time.monotonic() >= next_print:
            print(status_line(sim, tele, log), end="", flush=True)
            next_print = time.monotonic() + 0.25
        k = cv2.waitKey(1)
        if k != -1:
            tele.press(k & 0xFF)
        if cv2.getWindowProperty(win, cv2.WND_PROP_AUTOSIZE) < 0:   # window closed
            break
    cv2.destroyAllWindows()


def main():
    ap = add_sim_args(argparse.ArgumentParser(description=__doc__,
                                              formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--cv", action="store_true", help="OpenCV window instead of the MuJoCo viewer")
    args = ap.parse_args()
    sim, _ = session_from_args(args)
    print(f"environment {sim.environment_id}; standing up...")
    sim.settle()
    print(HELP)
    tele, log = KeyTeleop(), CollisionLog()
    try:
        (run_cv if args.cv else run_viewer)(sim, tele, log)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\ncollisions logged: {log.count}")
        sim.close()


if __name__ == "__main__":
    main()
