"""Obstacle memory (real_go2/obstacle_memory.py) and its use in the go-to-zone navigator. No robot needed."""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

from real_go2.goto_zone import ZoneNavigator  # noqa: E402
from real_go2.obstacle_memory import ObstacleMemory, describe_bearing, mem_cfg  # noqa: E402

COR = {"x_min": 0.05, "x_max": 1.3, "half_width_m": 0.3, "z_min": -0.20, "z_max": 0.45, "min_points": 3}


def box_points(x0, x1, y0, y1, z=0.1, step=0.03):
    """Points on the outline of a box (body or world frame, whichever the caller uses)."""
    xs, ys = np.arange(x0, x1 + 1e-9, step), np.arange(y0, y1 + 1e-9, step)
    pts = [(x, y0) for x in xs] + [(x, y1) for x in xs] + [(x0, y) for y in ys] + [(x1, y) for y in ys]
    return np.array([(x, y, z) for x, y in pts])


def test_object_is_remembered_after_it_leaves_the_view():
    m = ObstacleMemory()
    pose = (0.0, 0.0, 0.0)
    m.update_lidar(box_points(1.0, 1.2, -0.2, 0.2), pose, 0.0)       # seen 1 m ahead
    assert m.free_profile(pose, 24, COR, 2.5)[0] < 1.1
    # the dog walks up to 0.5 m and the box is now in the blind zone: the scan is empty
    pose = (0.5, 0.0, 0.0)
    for t in range(1, 30):
        m.update_lidar(np.zeros((0, 3)), pose, t * 0.1)
    assert m.free_profile(pose, 24, COR, 2.5)[0] < 0.6, "box forgotten while in the blind zone"
    # it turns 90 deg left: the box is now to its right, still remembered (odometry frame)
    pose = (0.5, 0.0, math.pi / 2)
    for t in range(30, 60):
        m.update_lidar(np.zeros((0, 3)), pose, t * 0.1)
    free = m.free_profile(pose, 24, COR, 2.5)
    assert free[18] < 0.6                                              # bin 18 = 270 deg = right
    d, b, _ = m.nearest(pose)
    assert d < 0.6 and describe_bearing(b) == "right"


def test_object_is_forgotten_when_seen_through():
    m = ObstacleMemory()
    pose = (0.0, 0.0, 0.0)
    m.update_lidar(box_points(1.0, 1.2, -0.2, 0.2), pose, 0.0)
    m.update_lidar(box_points(1.0, 1.2, -0.2, 0.2), pose, 0.1)
    # the box is moved away: the LiDAR now sees a wall 2.5 m ahead through where it was
    wall = box_points(2.5, 2.6, -1.0, 1.0)
    for t in range(2, 30):
        m.update_lidar(wall, pose, t * 0.1)
    assert m.free_profile(pose, 24, COR, 2.0)[0] >= 2.0 - 1e-6


def test_detection_without_lidar_hit_still_blocks():
    m = ObstacleMemory()
    pose = (1.0, 1.0, math.pi)
    obj = m.add_detection("chair", 0.8, pose, 0.0, xyz_body=(0.8, 0.0, 0.0))
    assert obj["label"] == "chair"
    assert np.allclose(obj["xy"], (0.2, 1.0), atol=1e-6)               # 0.8 m ahead while facing -x
    assert m.free_profile(pose, 24, COR, 2.5)[0] < 0.7
    near = m.objects_near(pose)
    assert near and near[0][0] == "chair" and abs(near[0][1] - 0.8) < 1e-6
    # a second sighting nearby updates the same object
    m.add_detection("chair", 0.7, pose, 1.0, xyz_body=(0.85, 0.05, 0.0))
    assert len(m.objects) == 1


def test_bearing_only_detection_labels_lidar_cluster():
    m = ObstacleMemory()
    pose = (0.0, 0.0, 0.0)
    m.update_lidar(box_points(1.5, 1.7, 0.4, 0.8), pose, 0.0)
    m.update_lidar(box_points(1.5, 1.7, 0.4, 0.8), pose, 0.1)
    obj = m.add_detection("backpack", 0.6, pose, 0.2, bearing=math.atan2(0.6, 1.5))
    assert obj is not None and abs(obj["xy"][0] - 1.5) < 0.1
    assert m.nearest(pose)[2] == "backpack"


def test_navigator_does_not_walk_into_a_remembered_object():
    cfg = yaml.safe_load((TEST2 / "real_go2/real_config.yaml").read_text())
    classes = cfg["classes"]
    nav = ZoneNavigator(classes, cfg["navigation"], have_lidar=True)
    nav.state, nav.target, nav.best_heading, nav._t0, nav.legs = "walk", 0, 0.0, 0.0, 1
    live = np.full(24, 2.5)                                            # the live LiDAR sees nothing ...
    mem = np.full(24, 2.5)
    mem[0] = 0.55                                                      # ... but the memory has a box 0.55 m ahead
    vx, wz = nav.update(0.1, 0.0, None, live, mem_free=mem)
    assert vx == 0.0 and nav.state in ("look", "backoff")
    assert "remembered" in nav.msg or nav.state == "backoff"


def test_navigator_backs_off_before_turning_next_to_an_object():
    cfg = yaml.safe_load((TEST2 / "real_go2/real_config.yaml").read_text())
    nav = ZoneNavigator(cfg["classes"], cfg["navigation"], have_lidar=True)
    nav.target = 0
    mem = np.full(24, 2.5)
    mem[6] = 0.30                                                      # something 0.30 m to the left
    nav.mem_free = mem
    nav._look(0.0, 0.0)
    assert nav.state == "backoff"
    vx, wz = nav.update(0.1, 0.0, None, None, mem_free=mem)
    assert abs(vx) < 1e-6 and wz == 0.0 and nav.cmd_vy < 0            # steps to the RIGHT, away from it
    mem2 = np.full(24, 2.5)                                            # clear now
    nav.update(0.2, 0.0, None, None, mem_free=mem2)
    assert nav.state == "look" and nav.cmd_vy == 0.0


def test_no_backoff_when_boxed_in():
    cfg = yaml.safe_load((TEST2 / "real_go2/real_config.yaml").read_text())
    nav = ZoneNavigator(cfg["classes"], cfg["navigation"], have_lidar=True)
    nav.target = 0
    nav.mem_free = np.full(24, 0.40)                                   # close on every side: nowhere to step
    nav._look(0.0, 0.0)
    assert nav.state == "look"


def test_mem_cfg_reads_real_config():
    cfg = yaml.safe_load((TEST2 / "real_go2/real_config.yaml").read_text())
    c = mem_cfg(cfg["navigation"])
    assert c["blind_m"] > 0 and c["forget_s"] > 0
