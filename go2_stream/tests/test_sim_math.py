import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception import depth_fusion as df  # noqa: E402

CAM = df.CameraModel.from_hfov(320, 180, 100.0)


def test_estimate_from_depth_flat_wall():
    depth = np.full((180, 320), 3.0, dtype=np.float32)
    est = df.estimate_from_depth(depth, (100, 60, 200, 120), CAM)
    xyz, rng, n = est
    assert xyz[2] == pytest.approx(3.0)
    assert xyz[0] == pytest.approx((150 - CAM.cx) / CAM.fx * 3.0)
    assert rng == pytest.approx(np.linalg.norm(xyz))
    assert n >= 20


def test_estimate_prefers_foreground_over_background():
    depth = np.full((180, 320), 6.0, dtype=np.float32)
    depth[70:110, 140:180] = 2.0                       # object occupies ~the centre of the box
    xyz, rng, n = df.estimate_from_depth(depth, (100, 50, 220, 130), CAM)
    assert xyz[2] == pytest.approx(2.0)


def test_estimate_ignores_invalid_pixels_and_returns_none_without_enough():
    depth = np.zeros((180, 320), dtype=np.float32)      # no returns
    assert df.estimate_from_depth(depth, (100, 60, 200, 120), CAM) is None
    depth[:] = np.inf
    assert df.estimate_from_depth(depth, (100, 60, 200, 120), CAM) is None
    depth[:] = 50.0                                      # beyond max_range
    assert df.estimate_from_depth(depth, (100, 60, 200, 120), CAM) is None


def test_estimate_clips_boxes_that_leave_the_image():
    depth = np.full((180, 320), 2.5, dtype=np.float32)
    assert df.estimate_from_depth(depth, (-50, -50, 40, 40), CAM) is not None
    assert df.estimate_from_depth(depth, (400, 10, 500, 60), CAM) is None   # completely outside


def test_optical_world_roundtrip_with_rotated_camera():
    # MuJoCo-style camera yawed 90 deg left, looking along world +y: right = +x(world)... build matrix explicitly
    cam_pos = np.array([1.0, 2.0, 0.3])
    right, up, back = np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0])   # looks along +y
    mat = np.stack([right, up, back], axis=1)
    p_world = np.array([[1.5, 5.0, 0.8]])
    p_opt = df.world_to_optical(p_world, cam_pos, mat)
    assert p_opt[0, 2] == pytest.approx(3.0)           # 3 m ahead along +y
    assert p_opt[0, 0] == pytest.approx(0.5)           # 0.5 m to the right (+x)
    assert p_opt[0, 1] == pytest.approx(-0.5)          # 0.5 m up -> optical y is down -> negative
    assert np.allclose(df.optical_to_world(p_opt, cam_pos, mat), p_world)
