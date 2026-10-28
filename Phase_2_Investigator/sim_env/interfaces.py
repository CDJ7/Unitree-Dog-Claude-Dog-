"""Data types shared between the simulation (Stage A) and the analysis (Stage B).

Everything downstream (detection, BPA, navigation, the GUI, the run recorder)
consumes these, whether they come from the live simulator, a recorded run or,
later, the real Go2. Conventions:

- World frame: the MuJoCo world, z up, metres, radians.
- Camera frame: OpenCV (x right, y down, z forward). `T_world_cam` maps camera
  coordinates to world coordinates. MuJoCo's own camera frame (y up, looking
  along -z) is converted once, in `sim_env.runtime`.
- Pixels: (u, v) = (column, row), pixel centres at integer coordinates, so a
  point X_cam projects to K @ X_cam / z.
- Depth: metres along the optical axis (z), not along the ray; `inf` where
  there is no return. Stored at the RGB resolution.
- LiDAR scan: ROS LaserScan convention in the LiDAR frame (x forward, angles
  counter-clockwise from x), `inf` where there is no return.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraFrame:
    t: float                       # simulation time, s
    rgb: np.ndarray                # uint8 (H, W, 3)
    depth_m: np.ndarray            # float32 (H, W), inf = no return
    K: np.ndarray                  # float64 (3, 3) intrinsics for rgb and depth_m
    T_world_cam: np.ndarray        # float64 (4, 4), OpenCV camera frame -> world
    frame_id: str = "front_rgbd"

    @property
    def size(self):
        return self.rgb.shape[1], self.rgb.shape[0]

    def backproject(self, u, v, depth=None):
        """World coordinates of pixel(s) (u, v) at their depth (or the given one)."""
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if depth is None:
            depth = self.depth_m[np.round(v).astype(int), np.round(u).astype(int)]
        z = np.asarray(depth, dtype=float)
        x = (u - self.K[0, 2]) / self.K[0, 0] * z
        y = (v - self.K[1, 2]) / self.K[1, 1] * z
        p = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        return (p @ self.T_world_cam.T)[..., :3]

    def project(self, points_world):
        """Pixel coordinates (N, 2) and depth (N,) of world points."""
        p = np.asarray(points_world, dtype=float).reshape(-1, 3)
        T_cam_world = np.linalg.inv(self.T_world_cam)
        pc = p @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (pc @ self.K.T)[:, :2] / z[:, None]
        return uv, z


@dataclass
class LidarScan:
    t: float
    ranges: np.ndarray             # float32 (N,), planar scan, inf = no return
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    points_lidar: np.ndarray       # float32 (M, 3), 3-D cloud in the LiDAR frame
    T_world_lidar: np.ndarray      # float64 (4, 4)
    frame_id: str = "mid360"

    @property
    def angles(self):
        return self.angle_min + self.angle_increment * np.arange(self.ranges.size)

    def scan_points_lidar(self):
        """(K, 2) xy points of the finite planar returns, LiDAR frame."""
        ok = np.isfinite(self.ranges)
        a = self.angles[ok]
        r = self.ranges[ok].astype(float)
        return np.column_stack([r * np.cos(a), r * np.sin(a)])

    def scan_points_world(self):
        xy = self.scan_points_lidar()
        p = np.column_stack([xy, np.zeros(len(xy))])
        return p @ self.T_world_lidar[:3, :3].T + self.T_world_lidar[:3, 3]

    def points_world(self):
        p = self.points_lidar.astype(float)
        return p @ self.T_world_lidar[:3, :3].T + self.T_world_lidar[:3, 3]


@dataclass
class ImuReading:
    t: float
    gyro: np.ndarray               # rad/s, IMU frame
    accel: np.ndarray              # m/s^2 specific force, IMU frame
    quat_wxyz: np.ndarray          # orientation of the IMU in the world


@dataclass
class RobotState:
    t: float
    position: np.ndarray           # base (x, y, z), world
    quat_wxyz: np.ndarray          # base orientation
    yaw: float                     # rad
    lin_vel_world: np.ndarray
    ang_vel_body: np.ndarray
    cmd_vel: np.ndarray            # last commanded (vx, vy, wz), body frame
    imu: ImuReading
    upright: float                 # z component of the base z axis (1 = level)

    @property
    def pose2d(self):
        return np.array([self.position[0], self.position[1], self.yaw])

    @property
    def T_world_base(self):
        return pose_matrix(self.position, quat_to_mat(self.quat_wxyz))


@dataclass
class Contact:
    t: float
    robot_geom: str
    other_geom: str
    other_body: str
    position: np.ndarray
    normal: np.ndarray
    force: float


@dataclass
class EventDescription:
    """What happened, as event playback (A5) consumes it.

    Built from ground truth in Stage A; Stage B fills the same fields from the
    analysis (source="analysis") so the same playback shows the inferred event.
    """
    mechanism: str                 # impact | castoff | projected | drip | transfer
    origin: np.ndarray             # (x, y, z) world
    weapon_category: str = "unknown"   # blunt | edged | none | unknown
    swing_path: np.ndarray | None = None   # (N, 3) weapon tip path, cast-off
    launches: list = field(default_factory=list)  # [{t, pos, vel, diameter}]
    confidence: float = 1.0
    source: str = "ground_truth"   # ground_truth | analysis
    notes: str = ""
    params: dict = field(default_factory=dict)       # mechanism parameters (axis, heading_deg, end, ...)
    description: str = ""


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def pose_matrix(position, rotation):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


def intrinsics_from_fovy(width, height, fovy_deg):
    """Pinhole K for a MuJoCo camera (square pixels, centred principal point)."""
    f = 0.5 * height / np.tan(np.radians(fovy_deg) / 2)
    return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.0]])
