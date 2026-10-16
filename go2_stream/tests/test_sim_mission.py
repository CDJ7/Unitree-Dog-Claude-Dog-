"""End to end in the MuJoCo sim: sim camera/depth -> detection thread -> Mission -> real Teleop -> sim robot.
Needs `mujoco` and a unitree_mujoco clone (skipped otherwise). Headless: uses the ray-cast render and ground-truth labels."""
import copy
import math
import os
import sys
import time

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("mujoco")
from sim import world as W  # noqa: E402
from sim.ground_truth import GroundTruthDetector  # noqa: E402
from mapping.sim_depth import make_sim_fuse_fn  # noqa: E402
from behavior.clearance import make_sim_clearance  # noqa: E402
from behavior.mission import Mission, State  # noqa: E402
from perception.pipeline import DetectionThread  # noqa: E402
from teleop import Teleop  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DCFG = yaml.safe_load(open(os.path.join(ROOT, "config", "detection.yaml")))
CTRL = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))["control"]
BCFG = yaml.safe_load(open(os.path.join(ROOT, "config", "behaviors.yaml")))


class Rig:
    def __init__(self, bcfg=None, **start):
        cfg = W.load_config()
        cfg["render"] = False
        cfg["camera"].update(width=160, height=90, fps=6)
        cfg["start_pose"].update(standing=True, **start)
        try:
            W.unitree_mujoco_dir(cfg)
        except FileNotFoundError:
            pytest.skip("unitree_mujoco clone not found (see sim/setup_sim.sh)")
        self.world = W.SimWorld(cfg)
        self.world.start()
        self.client = W.SimSportClient(self.world)
        self.teleop = Teleop(CTRL, self.client)
        self.teleop.start(use_keyboard=False)
        self.teleop.standing = True                       # the dog starts standing in these tests
        self.det = DetectionThread(GroundTruthDetector(self.world), self.world.buffer, DCFG,
                                   fuse_fn=make_sim_fuse_fn(self.world, DCFG["fusion"]))
        self.det.start()
        b = copy.deepcopy(bcfg or BCFG)
        b["safety"].update(settle_s=0.5, confirm_frames=3)
        b["actions"]["sit"]["hold_s"] = 0.5
        self.mission = Mission(b, self.det.latest, self.teleop, classes=["chair"],
                               clearance_fn=make_sim_clearance(self.world, b["safety"]["corridor"]))
        self.teleop.on_manual_input = self.mission.manual_input
        self.mission.start()

    def wait(self, cond, timeout):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if cond():
                return True
            time.sleep(0.1)
        return False

    def close(self):
        self.mission.stop()
        self.teleop.shutdown()
        self.det.stop()
        self.world.stop()


def chair_dist(world, name="obj_chair_3"):
    o = next(o for o in world.objects if o.name == name)
    x, y, _ = world.pose
    return math.hypot(o.center[0] - x, o.center[1] - y)


def test_dog_turns_walks_to_the_chair_stops_sits_and_rises():
    r = Rig(x=-2.2, y=1.2, yaw_deg=-25.0)                 # chair_3 is off to the left/right of straight ahead
    try:
        assert r.wait(lambda: r.mission.candidate is not None and r.mission._seen >= 3, 15), r.mission.status_text()
        ok, msg = r.mission.go()
        assert ok, msg
        assert r.wait(lambda: r.mission.state == State.DONE, 120), r.mission.status_text()
        stop = r.mission.table["chair"][1]
        d = chair_dist(r.world)
        assert stop - 0.1 <= d <= stop + 0.6, f"final distance {d:.2f} m, stop distance {stop} m"
        assert r.world.standing                           # sat, then rose again
    finally:
        r.close()


def test_pressing_a_key_during_the_approach_stops_the_dog():
    r = Rig(x=-2.2, y=0.0, yaw_deg=0.0)
    try:
        assert r.wait(lambda: r.mission.candidate is not None and r.mission._seen >= 3, 15)
        assert r.mission.go()[0]
        assert r.wait(lambda: r.world.pose[0] > -2.0, 30)          # it is walking
        r.teleop.key_down("space")
        assert r.mission.state == State.ABORTED
        time.sleep(0.4)
        p = r.world.pose
        time.sleep(0.8)
        q = r.world.pose
        assert math.hypot(q[0] - p[0], q[1] - p[1]) < 0.03     # stands still
    finally:
        r.close()


def test_a_large_obstacle_in_the_corridor_blocks_the_dog():
    """The corridor check sees a big surface (the east wall, ~5.3 m ahead) and the dog refuses to walk.
    The limits are widened for the test; thin obstacles (chair legs) are NOT reliably seen by this check."""
    b = copy.deepcopy(BCFG)
    b["safety"]["corridor"]["x_max"] = 6.0
    b["safety"]["corridor"]["stop_if_closer_m"] = 5.5
    b["safety"]["blocked_abort_s"] = 1.0
    r = Rig(bcfg=b, x=-2.2, y=0.0, yaw_deg=0.0)
    try:
        assert r.wait(lambda: r.mission.candidate is not None and r.mission._seen >= 3, 15)
        start = r.world.pose
        assert r.mission.go()[0]
        assert r.wait(lambda: r.mission.state == State.ABORTED, 20), r.mission.status_text()
        assert "blocked" in r.mission.message
        assert math.hypot(r.world.pose[0] - start[0], r.world.pose[1] - start[1]) < 0.2
    finally:
        r.close()
