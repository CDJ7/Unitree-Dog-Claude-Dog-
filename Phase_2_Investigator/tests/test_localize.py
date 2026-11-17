"""Relocalisation on a saved map: recover an unknown odometry offset from LiDAR points."""
import numpy as np

from mapping.localize import align
from navigation.occupancy import OccupancyGrid


def l_room():
    """An L-shaped room with a pillar: no rotational symmetry."""
    g = OccupancyGrid(origin=(-1, -1), size=(8.0, 6.0), res=0.05)
    g.seen[:] = True
    g.logodds[:] = -2.0
    walls = []
    for x in np.arange(0, 6, 0.02):
        walls += [(x, 0.0), (x, 4.0 if x < 3.5 else 2.5)]
    for y in np.arange(0, 4, 0.02):
        walls += [(0.0, y)]
        if y < 2.5:
            walls += [(6.0, y)]
    for x in np.arange(3.5, 6, 0.02):
        walls += [(x, 2.5)]
    for y in np.arange(2.5, 4, 0.02):
        walls += [(3.5, y)]
    for t in np.linspace(0, 2 * np.pi, 40):
        walls += [(2.0 + 0.2 * np.cos(t), 1.5 + 0.2 * np.sin(t))]
    walls = np.array(walls)
    r, c = g.to_cell(walls)
    g.logodds[r, c] = 3.0
    return g, walls


def test_recovers_offset():
    g, walls = l_room()
    rng = np.random.default_rng(1)
    robot_map = np.array([1.2, 1.0])
    vis = walls[np.linalg.norm(walls - robot_map, axis=1) < 4.0]
    vis = vis[rng.choice(len(vis), 600, replace=False)] + rng.normal(0, 0.02, (600, 2))
    # the same points in an odometry frame rotated by 70 deg and shifted
    yaw, t = np.radians(70), np.array([-3.0, 2.0])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    odom_pts = (vis - t) @ R                         # map = R @ odom + t  ->  odom = R^T (map - t)
    odom_robot = (robot_map - t) @ R
    res = align(g, odom_pts, odom_robot, 0.3)
    assert res["ok"], res
    assert abs(res["x"] - 1.2) < 0.1 and abs(res["y"] - 1.0) < 0.1, res
    assert abs(((res["dyaw"] - yaw + np.pi) % (2 * np.pi)) - np.pi) < np.radians(2), res
    assert abs(((res["yaw"] - (0.3 + yaw) + np.pi) % (2 * np.pi)) - np.pi) < np.radians(2), res


def test_rejects_a_different_room():
    g, _ = l_room()
    rng = np.random.default_rng(2)
    t = rng.uniform(0, 2 * np.pi, 500)
    circle = np.column_stack([3 * np.cos(t), 3 * np.sin(t)])     # a round room that is not on the map
    assert not align(g, circle)["ok"]
