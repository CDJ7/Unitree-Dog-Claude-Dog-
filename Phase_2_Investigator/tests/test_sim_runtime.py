"""A1: simulator session, sensors, interfaces and teleop mapping.

The simulator tests need the submodule (./scripts/setup.sh) and offscreen
rendering; they are skipped when the submodule is missing.
"""
import numpy as np
import pytest

from navigation.teleop import KeyTeleop
from navigation.velocity import SpeedLimits, VelocityInterface
from sim_env.interfaces import CameraFrame, intrinsics_from_fovy, pose_matrix
from sim_env.runtime import SIM_ROOT

needs_sim = pytest.mark.skipif(not (SIM_ROOT / "sim/native/controller.py").exists(),
                               reason="simulator submodule not fetched")


def test_project_backproject_roundtrip():
    K = intrinsics_from_fovy(320, 240, 60)
    assert np.isclose(K[1, 1], 120 / np.tan(np.radians(30)))
    R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # camera looking along world +x
    f = CameraFrame(0, np.zeros((240, 320, 3), np.uint8), np.full((240, 320), 2.0, np.float32), K,
                    pose_matrix([1, 2, 0.3], R))
    pts = f.backproject([10, 160, 300], [20, 120, 230])
    uv, z = f.project(pts)
    assert np.allclose(uv, [[10, 20], [160, 120], [300, 230]])
    assert np.allclose(z, 2.0)
    centre = f.backproject((320 - 1) / 2, (240 - 1) / 2)
    assert np.allclose(centre, [3, 2, 0.3])


def test_teleop_keys_latch_and_clip():
    t = KeyTeleop()
    for _ in range(10):
        t.press("w")
    assert np.isclose(t.target[0], 0.4)            # clipped to the gait's forward limit
    t.press(ord("A"))                               # GLFW code, upper case
    t.press("q")
    assert t.target[2] > 0 and t.target[1] > 0
    t.press(32)
    assert not t.target.any()
    t.press("r")
    assert t.reset_requested


def test_velocity_ramp_and_limits():
    sent = []

    class Rec(VelocityInterface):
        def _apply(self, cmd):
            sent.append(cmd)

    v = Rec(SpeedLimits(vx=0.3, vy=0.1, wz=0.5, accel=1.0))
    v.send(1.0, 1.0, -2.0, t=0.0)
    assert np.allclose(sent[-1], [0.3, 0.1, -0.5])
    v2 = Rec(SpeedLimits(accel=1.0))
    v2.send(0, 0, 0, t=0.0)
    v2.send(0.3, 0, 0, t=0.1)
    assert np.isclose(sent[-1][0], 0.1)            # ramped by accel * dt
    v2.send(-1, 0, 0, t=1.0)
    assert np.isclose(sent[-1][0], -0.15)          # backward limited to half


@pytest.fixture(scope="module")
def sim():
    from sim_env.runtime import SimSession
    s = SimSession("scenesmith_house_186")
    s.settle(1.5)
    yield s
    s.close()


@needs_sim
def test_stands_and_sensors(sim):
    st = sim.state()
    assert st.upright > 0.95 and 0.2 < st.position[2] < 0.4
    f = sim.camera()
    assert f.rgb.shape == (240, 320, 3) and f.rgb.std() > 5
    assert f.depth_m.shape == (240, 320) and np.isfinite(f.depth_m).mean() > 0.5
    # the camera looks forward along the robot's heading
    fwd = f.T_world_cam[:3, 2]
    assert fwd @ np.array([np.cos(st.yaw), np.sin(st.yaw), 0]) > 0.9
    scan = sim.lidar()
    assert np.isfinite(scan.ranges).sum() > 300 and len(scan.points_lidar) > 500
    assert 8.5 < np.linalg.norm(st.imu.accel) < 11


@needs_sim
def test_cmd_vel_expires(sim):
    sim.cmd_vel(0.3, 0, 0)
    sim.step(0.2)
    assert sim.active_cmd[0] > 0
    sim.step(sim.cmd_timeout + 0.1)                 # not re-sent: dead-man stops it
    assert not sim.active_cmd.any()
    sim.step(0.8)
    p0 = sim.state().position.copy()
    sim.step(0.5)
    assert np.linalg.norm(sim.state().position[:2] - p0[:2]) < 0.05


@needs_sim
def test_walking_into_a_wall_is_logged(sim):
    """Drive straight at the nearest obstacle ahead: contacts must be reported."""
    sim.reset()
    sim.settle(1.0)
    hits = []
    for _ in range(80):                              # up to 8 s at 0.3 m/s
        sim.cmd_vel(0.3, 0, 0)
        sim.step(0.1)
        hits += sim.contacts()
        if hits:
            break
    assert hits, "no contact reported after walking 2+ m forward"
    assert all(h.robot_geom not in ("FL", "FR", "RL", "RR") or abs(h.normal[2]) <= 0.7 for h in hits)


@needs_sim
def test_depth_has_no_multiray_holes():
    """At a doorway in house 186, mj_multiRay misses ~9% of floor pixels; we re-cast them."""
    from sim_env.runtime import SimSession
    s = SimSession("scenesmith_house_186", spawn=(4.53, 2.45, np.radians(-32)))
    try:
        s.settle(0.5)
        s.fix_depth_misses = False
        raw = s.camera().depth_m
        s.fix_depth_misses = True
        fixed = s.camera().depth_m
        holes = ~np.isfinite(raw)
        assert holes.mean() > 0.02, "pose no longer shows the mj_multiRay misses; test is moot"
        assert np.isfinite(fixed).mean() > 0.99
        assert np.all((fixed[holes] > 0.3) & (fixed[holes] < 3.0))     # nearby floor
        assert np.array_equal(fixed[~holes], raw[~holes])
    finally:
        s.close()
