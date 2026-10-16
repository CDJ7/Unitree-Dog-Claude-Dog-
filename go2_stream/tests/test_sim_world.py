"""Needs the `mujoco` package and a clone of unitree_mujoco (UNITREE_MUJOCO or ~/unitree_mujoco); skipped otherwise."""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
mujoco = pytest.importorskip("mujoco")
from sim import world as W  # noqa: E402


def _cfg(**start):
    cfg = W.load_config()
    cfg["render"] = False
    cfg["camera"].update(width=96, height=54, fps=10)
    cfg["start_pose"].update(start)
    try:
        W.unitree_mujoco_dir(cfg)
    except FileNotFoundError:
        pytest.skip("unitree_mujoco clone not found (see sim/setup_sim.sh)")
    return cfg


@pytest.fixture
def standing_world():
    w = W.SimWorld(_cfg(standing=True))
    w.start()
    time.sleep(0.3)
    yield w
    w.stop()


def test_labelled_objects_are_parsed_from_the_scene(standing_world):
    classes = sorted(o.cls for o in standing_world.objects)
    assert "chair" in classes and "dining table" in classes and "backpack" in classes and "bottle" in classes


def test_move_refused_while_lying_then_accepted():
    w = W.SimWorld(_cfg(standing=False))
    w.start()
    c = W.SimSportClient(w)
    assert c.Move(0.3, 0, 0) != 0
    c.StandUp()
    t = time.monotonic()
    while not w.standing and time.monotonic() - t < 4:
        time.sleep(0.05)
    assert w.standing and c.Move(0.3, 0, 0) == 0
    w.stop()


def test_velocity_integrates_and_lapses_without_refresh(standing_world):
    w, c = standing_world, W.SimSportClient(standing_world)
    x0 = w.pose[0]
    t = time.monotonic()
    while time.monotonic() - t < 1.0:
        c.Move(0.4, 0, 0)
        time.sleep(0.03)
    assert w.pose[0] - x0 == pytest.approx(0.4, abs=0.08)
    time.sleep(0.8)                                    # command lapses (command_timeout_s = 0.5)
    p = w.pose
    time.sleep(0.3)
    assert w.pose == p


def test_furniture_blocks_the_robot():
    w = W.SimWorld(_cfg(standing=True, x=0.8, y=1.5, yaw_deg=0.0))   # facing the cardboard box at x=2.0
    w.start()
    c = W.SimSportClient(w)
    t = time.monotonic()
    while time.monotonic() - t < 4:
        c.Move(0.4, 0, 0)
        time.sleep(0.03)
    assert w.pose[0] < 1.6
    w.stop()


def test_frames_depth_and_pose_are_published_together(standing_world):
    time.sleep(0.5)
    frame, seq, _ = standing_world.buffer.get()
    b = standing_world.bundle_for(frame)
    assert b is not None and b.seq == seq and b.depth.shape == frame.shape[:2]
    assert (b.depth > 0).any()
