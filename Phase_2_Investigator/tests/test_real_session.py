"""Real-Go2 session pieces that need no robot: planar scan from the deskewed cloud, frames, stopping."""
import threading

import numpy as np
import pytest

from real_robot.session import OperatorStop, RealGo2Session, cloud_to_scan


def test_cloud_to_scan_takes_the_band_and_the_nearest_point():
    pts = np.array([[2.0, 0.0, 0.3],     # ahead, in the band
                    [1.5, 0.0, 0.02],    # floor: ignored
                    [3.0, 0.0, 0.3],     # behind the first: not the nearest
                    [0.0, 1.0, 1.5],     # above the band: ignored
                    [0.0, -1.0, 0.3]])   # right side
    r, amin, ainc = cloud_to_scan(pts, (0.0, 0.0), 0.0, n=360, gap=0)
    ang = amin + ainc * np.arange(360)
    assert abs(r[np.argmin(np.abs(ang))] - 2.0) < 1e-6
    assert abs(r[np.argmin(np.abs(ang + np.pi / 2))] - 1.0) < 1e-6
    assert np.isnan(r[np.argmin(np.abs(ang - np.pi / 2))])               # no point there: no information
    r2, _, _ = cloud_to_scan(pts, (0.0, 0.0), np.pi / 2, n=360, gap=0)      # robot facing +y: the wall at -y is behind
    assert abs(r2[np.argmin(np.abs(np.abs(ang) - np.pi))] - 1.0) < 1e-6


class FakeLink:
    """A dog standing at odom (1, 2) facing 90 deg, in a 4 x 4 m box, floor at odom z = -0.3."""
    name = "fake"

    def __init__(self):
        self.t, self.manual, self.moves = 0.0, threading.Event(), []
        side = np.linspace(-2, 2, 81)
        walls = [np.column_stack([side, np.full(81, v)]) for v in (-2, 2)]
        walls += [np.column_stack([np.full(81, v), side]) for v in (-2, 2)]
        w = np.vstack(walls) + [1.0, 2.0]
        self.pts = np.column_stack([w, np.full(len(w), 0.0)])          # 0.3 m above the floor
        floor = np.column_stack([np.random.default_rng(0).uniform(-1, 3, (200, 2)), np.full(200, -0.3)])
        self.pts = np.vstack([self.pts, floor])

    def now(self): return self.t
    def pose(self): return (1.0, 2.0, 0.01, np.pi / 2, self.t)
    def cloud(self): return self.pts.copy(), self.t
    def frame(self): return np.zeros((720, 1280, 3), np.uint8), self.t
    def move(self, vx, vy, wz): self.moves.append((vx, vy, wz))
    def halt(self): self.moves.append((0, 0, 0))
    def advance(self, dt): self.t += dt
    def takeover(self): return self.manual.is_set()
    def close(self): pass


def test_session_frames_floor_and_map_transform():
    s = RealGo2Session(FakeLink(), keys=False)
    st = s.state()
    assert abs(st.position[2] - 0.31) < 1e-6                        # body 0.31 m above the floor
    scan = s.lidar()
    pw = scan.points_world()
    assert pw[:, 2].min() >= 0.12                                    # floor points removed
    assert np.isfinite(scan.ranges).mean() > 0.9                     # walls all round
    ahead = scan.ranges[np.argmin(np.abs(scan.angles))]
    assert abs(ahead - 2.0) < 0.06                                   # facing +y, wall 2 m away
    s.set_map_transform(10.0, 0.0, np.pi)                            # map = rotate 180 deg, then +10 m in x
    st = s.state()
    assert np.allclose(st.position[:2], [9.0, -2.0]) and abs(abs(st.yaw) - np.pi / 2) < 1e-6
    T = s.camera_pose()
    assert np.allclose(T[:3, 2], [0, -1, 0], atol=1e-6)              # camera looks along the map's -y now
    assert s.camera().rgb.shape == (720, 1280, 3) and len(s.depth_points()) == 0


def test_takeover_stops_the_run():
    link = FakeLink()
    s = RealGo2Session(link, keys=False)
    s.cmd_vel(0.2, 0, 0)
    s.step(0.1)
    link.manual.set()
    with pytest.raises(OperatorStop):
        s.step(0.1)
    assert link.moves[-1] == (0, 0, 0)
