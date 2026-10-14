import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from perception import depth_fusion as df  # noqa: E402

CAM = df.CameraModel(width=1280, height=720, fx=640.0, fy=640.0, cx=640.0, cy=360.0)


def test_hfov_intrinsics():
    cam = df.CameraModel.from_hfov(1280, 720, 90.0)
    assert cam.fx == pytest.approx(640.0) and cam.cx == 640.0 and cam.cy == 360.0


def test_config_intrinsics_scale_with_frame_size():
    cfg = {"fx": 640, "fy": 640, "cx": 640, "cy": 360, "ref_width": 1280, "ref_height": 720}
    cam = df.CameraModel.from_config(cfg, 640, 360)
    assert (cam.fx, cam.cx, cam.cy) == (320.0, 320.0, 180.0)


def test_body_to_camera_axes():
    p = df.body_to_camera([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(p[0], [0, 0, 2])        # ahead -> +z (optical)
    assert np.allclose(p[1], [-1, 0, 0])       # left  -> -x (optical, x is right)
    assert np.allclose(p[2], [0, -1, 0])       # up    -> -y (optical, y is down)


def test_body_to_camera_translation():
    p = df.body_to_camera([[1.0, 0, 0.1]], translation=(0.3, 0, 0.1))
    assert np.allclose(p[0], [0, 0, 0.7])


def test_camera_pitch_down_moves_horizon_object_up_in_image():
    p = df.body_to_camera([[3.0, 0, 0]], pitch_deg=10.0)
    u, v, _ = df.project_points(p, CAM)
    assert v[0] < CAM.cy                      # above image centre
    assert u[0] == pytest.approx(CAM.cx)


def test_project_then_deproject_roundtrip():
    pt = np.array([[0.5, -0.2, 3.0]])
    u, v, idx = df.project_points(pt, CAM)
    back = df.deproject(u[0], v[0], 3.0, CAM)
    assert np.allclose(back, pt[0])
    assert idx.tolist() == [0]


def test_project_drops_behind_camera_nan_and_out_of_frame():
    pts = np.array([[0, 0, -1.0], [np.nan, 0, 2.0], [100.0, 0, 1.0], [0, 0, 2.0]])
    u, v, idx = df.project_points(pts, CAM)
    assert idx.tolist() == [3]


def test_odom_to_body_yaw_90():
    q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]          # robot yawed +90 deg (facing odom +y)
    p = df.odom_to_body([[1.0, 3.0, 0.0]], position=[1.0, 1.0, 0.0], q_wxyz=q)
    assert np.allclose(p[0], [2.0, 0.0, 0.0], atol=1e-9)        # 2 m straight ahead of the robot


def _box_scene(front_range, back_range, n=40, rng=np.random.default_rng(0)):
    """Points inside a box: half the points on the object, half on a background wall behind it."""
    u = rng.uniform(450, 550, n)
    v = rng.uniform(250, 350, n)
    r = np.where(np.arange(n) % 2 == 0, front_range, back_range).astype(float)
    return u, v, r


def test_select_ignores_background_and_returns_foreground():
    u, v, r = _box_scene(2.0, 6.0)
    sel = df.select_box_points(u, v, r, (400, 200, 600, 400))
    assert sel is not None
    assert np.allclose(r[sel], 2.0)


def test_select_ignores_zero_nan_and_out_of_range():
    u = np.full(6, 500.0)
    v = np.full(6, 300.0)
    r = np.array([0.0, np.nan, 99.0, 0.05, 2.0, 2.1])
    sel = df.select_box_points(u, v, r, (400, 200, 600, 400), min_points=2)
    assert sorted(r[sel].tolist()) == [2.0, 2.1]


def test_select_returns_none_with_too_few_points():
    u = np.array([500.0, 500.0])
    v = np.array([300.0, 300.0])
    assert df.select_box_points(u, v, np.array([2.0, 2.0]), (400, 200, 600, 400), min_points=3) is None
    assert df.select_box_points(np.empty(0), np.empty(0), np.empty(0), (0, 0, 10, 10)) is None


def test_select_falls_back_to_full_box_when_centre_is_empty():
    u = np.array([410.0, 420.0, 590.0, 580.0])        # only near the box edges
    v = np.array([210.0, 390.0, 210.0, 390.0])
    r = np.full(4, 3.0)
    assert df.select_box_points(u, v, r, (400, 200, 600, 400), central_frac=0.3) is not None


def test_summarize_points_median_is_robust():
    pts = np.array([[0, 0, 2.0], [0, 0, 2.0], [0, 0, 2.0], [0, 0, 50.0]])
    xyz, dist = df.summarize_points(pts)
    assert dist == pytest.approx(2.0)


def test_end_to_end_distance_of_object_at_known_range():
    """Object 2 m ahead of the robot: LiDAR points on it should give distance ~2 m, xyz_body ~ (2,0,0.3)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from perception import pipeline
    from perception.detector import Detection
    rng = np.random.default_rng(1)
    obj = np.column_stack([np.full(60, 2.0), rng.uniform(-0.2, 0.2, 60), rng.uniform(0.1, 0.5, 60)])
    wall = np.column_stack([np.full(200, 6.0), rng.uniform(-3, 3, 200), rng.uniform(0, 2, 200)])
    pts = np.vstack([obj, wall]).astype("<f4")
    rec = np.zeros(len(pts), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    f = lambda n, o: NS(name=n, offset=o, datatype=7, count=1)  # noqa: E731
    msg = NS(is_bigendian=False, point_step=12, width=len(pts), height=1, data=list(rec.tobytes()),
             fields=[f("x", 0), f("y", 4), f("z", 8)], header=NS(frame_id="base_link"))
    cfg = {"camera_model": {"hfov_deg": 90.0}, "extrinsics": {"translation": [0, 0, 0], "pitch_deg": 0, "yaw_deg": 0, "roll_deg": 0},
           "fusion": {"central_frac": 0.4, "min_points": 3, "foreground_frac": 0.5, "min_range_m": 0.2,
                      "max_range_m": 20.0, "lidar_frame": "auto"}}
    # object spans y in [-0.2,0.2] at x=2 -> u in 640 +/- 128*... ; z in [0.1,0.5] -> v above centre
    box = (500, 200, 780, 400)
    miss = (0, 0, 50, 50)
    import lidar as lidar_mod  # uses numpy + cv2 only at import time
    dets, uvr, status = pipeline.fuse_detections(
        [Detection(0, "person", 0.9, box), Detection(0, "chair", 0.8, miss)],
        (720, 1280, 3), msg, None, cfg, lidar_mod)
    assert dets[0].distance_m == pytest.approx(2.0, abs=0.25)
    assert dets[0].xyz_body[0] == pytest.approx(2.0, abs=0.05)
    assert dets[1].distance_m is None                          # no points there -> "depth: n/a"
    assert uvr is not None and len(uvr) > 0
