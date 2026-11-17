"""The real Unitree Go2 behind the same interface as the simulator's SimSession.

Phase 2's navigation (navigation/explorer.py: occupancy map, frontier exploration, planner, safety
layer) and the zone mapper only call these methods of a "session":
    time, state(), lidar(), camera(), camera_pose(), depth_points(), contacts(), cmd_vel(), stop(), step(dt)
`RealGo2Session` provides them from a *link*:

  Go2Link   the real dog over Ethernet (Unitree SDK2 / DDS), reusing Test 1's go2_stream modules:
            camera (VideoClient), rt/sportmodestate (odometry pose), the LiDAR on rt/utlidar/cloud_deskewed,
            and Teleop for motion (speed caps, 0.3 s dead-man, StopMove on exit).
  SimLink   the MuJoCo simulator presented exactly like the real dog (pose, an odometry-frame point cloud,
            camera frames, velocity commands), so this whole real-robot code path is tested in simulation.

The LiDAR: rt/utlidar/cloud is in the sensor's own frame and the L1 is mounted upside down (Test 2 found
that the floor then looks like an obstacle 5 cm away). rt/utlidar/cloud_deskewed is in the odometry frame,
the same frame as the sportmodestate pose, with the dog's own body removed. Here its points are shifted so
the floor is at z = 0 (as in the simulator), and the planar scan the occupancy grid needs is made from the
points between `scan_band` heights. The cloud is accumulated over `cloud_window_s`, because one L1 message
covers only part of the circle.

Stopping: the run ends (raises OperatorStop, the map is saved) when you press any key in the terminal
(SPACE, Enter, q, ...), when the Unitree remote or a Test 1 teleop key takes over, or when the pose or the
LiDAR goes stale for more than `stale_s`. Every exit path ends with StopMove (Test 1 teleop).
"""
from __future__ import annotations

import logging
import os
import select
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from sim_env.interfaces import CameraFrame, ImuReading, LidarScan, RobotState, intrinsics_from_fovy, pose_matrix

PHASE2 = Path(__file__).resolve().parents[1]
REPO = PHASE2.parent
GO2_STREAM = REPO / "go2_stream"
log = logging.getLogger("real_go2")


def real_safety_config():
    """Phase 2's safety layer measures clearance from the dog's nose (half length 0.35 m ahead of the centre).
    The simulator values (stop 0.20 m, slow from 0.60 m) are too tight for the real dog: in the Test 2 lab runs
    (13 Nov 2026) a 0.50 m stop measured from the centre left ~15 cm and the dog touched obstacles, which was
    fixed with 0.60 m + 0.5 s x speed from the centre (0.75 m at 0.3 m/s, a 0.40 m nose gap).
    Same margins here: stop at a 0.40 m nose gap, slow from 0.90 m; side and turning margins also larger."""
    from navigation.local import SafetyConfig
    return SafetyConfig(stop_dist=0.40, slow_dist=0.90, corridor_margin=0.08, sweep_radius=0.48)


class OperatorStop(KeyboardInterrupt):
    """The operator (terminal key, remote, teleop key) or a safety check ended the autonomous run."""


def add_real_args(ap):
    g = ap.add_argument_group("real robot (--backend real)")
    g.add_argument("--link", choices=["go2", "sim"], default="go2",
                   help="go2 = the dog over Ethernet; sim = the simulator presented like the dog (testing)")
    g.add_argument("--iface", help="Ethernet interface on 192.168.123.x (default: auto-detect)")
    g.add_argument("--max-speed", type=float, default=0.30, help="forward speed cap, m/s (navigation uses <= 0.3)")
    g.add_argument("--no-stand", action="store_true", help="do not send StandUp at the start")
    g.add_argument("--body-height", type=float, default=0.31,
                   help="floor distance below the body origin, m (scripts/real_check.py measures it)")
    return ap


# ------------------------------------------------------------------------------------------ helpers
def yaw_to_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def cloud_to_scan(points, origin_xy, yaw, n=360, band=(0.12, 0.60), range_min=0.35, range_max=6.0, gap=2):
    """Planar scan (n ranges over 360 deg, robot frame, angle 0 = straight ahead) from world points
    (floor at z = 0): per angle bin, the nearest point between band[0] and band[1] high.
    Bins with no point are NaN ("no information", skipped by the occupancy grid), not inf: a sparse cloud
    must not carve free space through walls. Holes of up to `gap` bins between returns are filled with the
    larger neighbouring range."""
    p = np.asarray(points, float)
    sel = (p[:, 2] >= band[0]) & (p[:, 2] <= band[1]) if len(p) else np.zeros(0, bool)
    d = p[sel, :2] - np.asarray(origin_xy, float)
    r = np.hypot(d[:, 0], d[:, 1])
    a = (np.arctan2(d[:, 1], d[:, 0]) - yaw + np.pi) % (2 * np.pi) - np.pi
    ok = (r >= range_min) & (r <= range_max)
    ranges = np.full(n, np.inf, np.float32)
    if ok.any():
        b = np.floor((a[ok] + np.pi) / (2 * np.pi) * n).astype(int) % n
        np.minimum.at(ranges, b, r[ok].astype(np.float32))
    ranges[np.isinf(ranges)] = np.nan
    if gap > 0 and np.isfinite(ranges).any():
        filled = ranges.copy()
        for i in np.flatnonzero(np.isnan(ranges)):
            for k in range(1, gap + 1):
                lo, hi = ranges[(i - k) % n], ranges[(i + k) % n]
                if np.isfinite(lo) and np.isfinite(hi):
                    filled[i] = max(lo, hi)
                    break
        ranges = filled
    return ranges, -np.pi, 2 * np.pi / n


class TerminalKeys:
    """Any key pressed in this terminal sets `pressed` (works over SSH and on Wayland, unlike pynput)."""

    def __init__(self):
        self.pressed = threading.Event()
        self.key = None
        self._stop = threading.Event()
        self._old = None
        if not sys.stdin.isatty():
            return
        import termios
        import tty
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        threading.Thread(target=self._run, daemon=True, name="terminal-keys").start()

    def _run(self):
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if r:
                self.key = os.read(self._fd, 1)
                self.pressed.set()

    def close(self):
        self._stop.set()
        if self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._old = None


# --------------------------------------------------------------------------------------------- links
class Go2Link:
    """The real Go2 over Ethernet via Test 1's go2_stream modules (imported, not changed)."""

    name = "go2"

    def __init__(self, iface=None, max_speed=0.30, stand=True, lidar_topic="rt/utlidar/cloud_deskewed", control=True):
        import yaml
        if str(GO2_STREAM) not in sys.path:
            sys.path.append(str(GO2_STREAM))           # Test 1 modules are flat: camera, lidar, teleop
        cfg = yaml.safe_load((GO2_STREAM / "config.yaml").read_text())
        iface = pick_iface(iface or cfg["network"]["iface"])
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        log.info("DDS on %s", iface)
        ChannelFactoryInitialize(0, iface)
        import camera as g2cam
        import lidar as g2lidar
        import teleop as g2tele
        self._lidar_mod = g2lidar
        self.camera = g2cam.CameraThread(cfg["camera"], iface)
        self.camera.start()
        self.state = g2lidar.StateSource(cfg["state"])
        lcfg = dict(cfg["lidar"], topic=lidar_topic)       # Test 1's config file is not changed
        self.lidar = g2lidar.LidarSource(lcfg)
        c = cfg["control"]
        self.manual = threading.Event()
        self._cloud_seq = -1
        self._warned_frame = False
        self.tele = None
        if not control:                                    # sensors only: no SportClient, nothing can move
            return
        self.tele = g2tele.Teleop(c, max_speed=None)
        self.tele.limits = (min(max_speed, c["hard_cap_vx"]), min(0.15, c["hard_cap_vy"]), min(0.8, c["hard_cap_vyaw"]))
        self.tele.on_manual_input = lambda key: self.manual.set()
        self.tele.start(use_keyboard=True)                 # Test 1 keys still work where pynput can see them
        if stand:
            self.tele.request_action("stand")
            time.sleep(3.0)

    def now(self):
        return time.monotonic()

    def pose(self):
        """(x, y, z, yaw, stamp) in the odometry frame, or None."""
        s, seq, stamp = self.state.sport.get()
        if s is None:
            return None
        return (float(s.position[0]), float(s.position[1]), float(s.position[2]), float(s.imu_state.rpy[2]), stamp)

    def cloud(self):
        """(points (N, 3) odometry frame, stamp) of the newest cloud not yet returned, else None."""
        msg, seq, stamp = self.lidar.latest.get()
        if msg is None or seq == self._cloud_seq:
            return None
        self._cloud_seq = seq
        frame_id = str(getattr(msg.header, "frame_id", ""))
        if "odom" not in frame_id.lower() and not self._warned_frame:
            log.error("LiDAR cloud frame is '%s', not odom: use rt/utlidar/cloud_deskewed.", frame_id)
            self._warned_frame = True
        return self._lidar_mod.pointcloud2_to_xyz(msg).astype(float), stamp

    def frame(self):
        """(RGB image, stamp) or None."""
        import cv2
        img, seq, stamp = self.camera.buffer.get()
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), stamp

    def move(self, vx, vy, wz):
        if self.tele is not None:
            self.tele.set_autonomy(vx, vy, wz)

    def halt(self):
        if self.tele is not None:
            self.tele.clear_autonomy()

    def advance(self, dt):
        time.sleep(max(0.0, dt))

    def takeover(self):
        return self.manual.is_set() or (self.tele is not None and self.tele.quit_event.is_set())

    def close(self):
        if self.tele is not None:
            self.tele.clear_autonomy()
            self.tele.shutdown()                            # always ends with StopMove
        self.camera.stop()


def pick_iface(iface):
    """The interface with a 192.168.123.x address (auto-detected when there is exactly one)."""
    import subprocess
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        out = ""
    found = [ln.split()[1] for ln in out.splitlines() if len(ln.split()) > 3 and ln.split()[3].startswith("192.168.123.")]
    if iface in found:
        return iface
    if len(found) == 1:
        log.warning("Using %s (the interface on 192.168.123.x) instead of %s", found[0], iface)
        return found[0]
    raise SystemExit(f"No interface with a 192.168.123.x address (found: {found or 'none'}). Set the static IP "
                     "(real_robot/README.md, step 2) and check `ip -br addr`.")


class SimLink:
    """The simulator, seen through the same narrow channel as the real dog (for tests and dry runs).
    The 'odometry frame' is the sim world; floor_z_odom shifts it like the real odometry frame (floor not at 0)."""

    name = "sim"

    def __init__(self, sim, floor_z_odom=-0.02, odom_offset=(0.0, 0.0, 0.0)):
        """odom_offset (dx, dy, dyaw): where the sim world origin lies in this fake 'odometry' frame, like a
        real dog that booted somewhere else (tests relocalisation)."""
        self.sim, self.dz = sim, float(floor_z_odom)
        self.off = np.asarray(odom_offset, float)
        self._t_cloud = -1.0
        self.manual = threading.Event()

    def _to_odom(self, xy):
        c, s = np.cos(self.off[2]), np.sin(self.off[2])
        return np.asarray(xy, float) @ np.array([[c, s], [-s, c]]) + self.off[:2]

    def now(self):
        return self.sim.time

    def pose(self):
        st = self.sim.state()
        x, y = self._to_odom(st.position[:2])
        yaw = (st.yaw + self.off[2] + np.pi) % (2 * np.pi) - np.pi
        return (float(x), float(y), float(st.position[2]) + self.dz, float(yaw), st.t)

    def cloud(self):
        if self.sim.time - self._t_cloud < 0.099:
            return None
        self._t_cloud = self.sim.time
        scan = self.sim.lidar()
        pw = np.vstack([scan.points_world(), scan.scan_points_world()])
        pw[:, 2] += self.dz
        pw[:, :2] = self._to_odom(pw[:, :2])
        return pw, self.sim.time

    def frame(self):
        return self.sim.camera().rgb, self.sim.time

    def move(self, vx, vy, wz):
        self.sim.cmd_vel(vx, vy, wz)

    def halt(self):
        self.sim.cmd_vel(0, 0, 0)

    def advance(self, dt):
        self.sim.step(dt)

    def takeover(self):
        return self.manual.is_set()

    def close(self):
        if hasattr(self.sim, "close"):
            self.sim.close()


# -------------------------------------------------------------------------------------------- session
class RealGo2Session:
    """SimSession look-alike over a link (Go2Link or SimLink)."""

    def __init__(self, link, body_height=0.31, scan_band=(0.12, 0.60), cloud_window_s=0.5, stale_s=1.0,
                 camera_fovy_deg=56.0, camera_xyz=(0.30, 0.0, 0.05), keys=True):
        self.link = link
        self.body_height, self.scan_band = float(body_height), scan_band
        self.cloud_window_s, self.stale_s = float(cloud_window_s), float(stale_s)
        self.camera_fovy_deg, self.camera_xyz = float(camera_fovy_deg), np.asarray(camera_xyz, float)
        self.environment_id = f"real_{link.name}"
        self.safety_config = real_safety_config()      # used by the scripts in place of the simulator values
        self._clouds = deque()
        self._floor = None
        self._cmd = np.zeros(3)
        self._t0 = link.now()
        self._next = 0.0
        self.tf = np.zeros(3)                  # odometry frame -> map frame (dx, dy, dyaw), set by relocalisation
        self.keys = TerminalKeys() if keys else None
        self._wait_for_data()

    # ------------------------------------------------------------ frames
    def set_map_transform(self, dx, dy, dyaw):
        self.tf = np.array([dx, dy, dyaw], float)

    def _xy_to_map(self, xy):
        c, s = np.cos(self.tf[2]), np.sin(self.tf[2])
        return np.asarray(xy, float) @ np.array([[c, s], [-s, c]]) + self.tf[:2]

    def _pose(self):
        """(x, y, z, yaw, stamp) in the map frame."""
        x, y, z, yaw, stamp = self.link.pose()
        mx, my = self._xy_to_map((x, y))
        return float(mx), float(my), z, float((yaw + self.tf[2] + np.pi) % (2 * np.pi) - np.pi), stamp

    def odom_points(self):
        """Obstacle points (N, 3) of the recent clouds in the *odometry* frame, floor at z = 0 (relocalisation)."""
        self._pull_cloud()
        z = self.link.pose()[2]
        pts = np.vstack([c[0] for c in self._clouds]) if self._clouds else np.zeros((0, 3))
        pts = pts.copy()
        pts[:, 2] -= self._floor_z(z)
        return pts[(pts[:, 2] >= self.scan_band[0]) & (pts[:, 2] <= self.scan_band[1])]

    @classmethod
    def from_args(cls, a, sim=None):
        if a.link == "sim":
            if sim is None:
                from sim_env.launch import session_from_args
                if not (getattr(a, "scenario", None) or getattr(a, "random", None) is not None):
                    a.random = 1854
                if getattr(a, "random", None) is not None and not getattr(a, "env", None):
                    from sim_env.scenarios.random_sim import write_random
                    a.scenario = str(write_random(a.random, env=getattr(a, "house", "scenesmith_house_185")))
                    a.random = None
                sim, _ = session_from_args(a)
            return cls(SimLink(sim), body_height=getattr(a, "body_height", 0.31))
        print("REAL ROBOT: the dog will move on its own. Clear the area, keep the remote in hand.\n"
              "Press any key in this terminal (or use the remote) to stop it.", flush=True)
        return cls(Go2Link(iface=a.iface, max_speed=a.max_speed, stand=not a.no_stand),
                   body_height=getattr(a, "body_height", 0.31))

    # ------------------------------------------------------------- timing
    @property
    def time(self):
        return float(self.link.now() - self._t0)

    def _wait_for_data(self, timeout=15.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self._pull_cloud()
            if self.link.pose() is not None and self._clouds:
                return
            self.link.advance(0.05)
        raise SystemExit("No pose or LiDAR data from the dog after 15 s: run Test 1's preflight.py and check the "
                         "LiDAR topic (rt/utlidar/cloud_deskewed).")

    def step(self, seconds=0.1):
        """Keep the command alive and wait (real time) or simulate (SimLink) `seconds`."""
        if self.keys is not None and self.keys.pressed.is_set():
            self.halt_now()
            raise OperatorStop(f"key {self.keys.key!r} pressed in the terminal")
        if self.link.takeover():
            self.halt_now()
            raise OperatorStop("manual control took over")
        if np.any(self._cmd):
            self.link.move(*self._cmd)
        if self.link.name == "sim":
            self.link.advance(seconds)
        else:
            self._next = max(self._next + seconds, self.time)
            self.link.advance(self._next - self.time)
        p = self.link.pose()
        if p is None or self.link.now() - p[4] > self.stale_s:
            self.halt_now()
            raise OperatorStop("odometry pose is stale: stopped for safety")

    def halt_now(self):
        self._cmd[:] = 0
        self.link.halt()

    # ------------------------------------------------------------ control
    def cmd_vel(self, vx=0.0, vy=0.0, wz=0.0):
        self._cmd[:] = (vx, vy, wz)
        self.link.move(vx, vy, wz)

    def stop(self):
        self.halt_now()

    # ------------------------------------------------------------ sensors
    def state(self) -> RobotState:
        x, y, z, yaw, stamp = self._pose()
        floor = self._floor_z(z)
        q = yaw_to_quat(yaw)
        imu = ImuReading(t=self.time, gyro=np.zeros(3), accel=np.array([0, 0, 9.81]), quat_wxyz=q)
        return RobotState(t=self.time, position=np.array([x, y, z - floor]), quat_wxyz=q, yaw=yaw,
                          lin_vel_world=np.zeros(3), ang_vel_body=np.zeros(3), cmd_vel=self._cmd.copy(),
                          imu=imu, upright=1.0)

    def _floor_z(self, body_z):
        """Floor height in the odometry frame: the body height below the base, refined from the cloud."""
        guess = body_z - self.body_height
        if self._floor is None:
            self._floor = guess
        return self._floor

    def _pull_cloud(self):
        c = self.link.cloud()
        if c is not None:
            self._clouds.append(c)
        now = self.link.now()
        while self._clouds and now - self._clouds[0][1] > self.cloud_window_s:
            self._clouds.popleft()

    def lidar(self) -> LidarScan:
        self._pull_cloud()
        x, y, z, yaw, _ = self._pose()
        floor = self._floor_z(z)
        pts = np.vstack([c[0] for c in self._clouds]) if self._clouds else np.zeros((0, 3))
        pts = pts.copy()
        pts[:, 2] -= floor
        if np.any(self.tf):
            pts[:, :2] = self._xy_to_map(pts[:, :2])
        ranges, amin, ainc = cloud_to_scan(pts, (x, y), yaw, band=self.scan_band)
        lz = self.scan_band[0] / 2 + self.scan_band[1] / 2
        T = pose_matrix(np.array([x, y, lz]), pose_matrix(np.zeros(3), np.eye(3))[:3, :3])
        c, s = np.cos(yaw), np.sin(yaw)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        # keep obstacle points only: above the floor (the occupancy grid's low layer and the safety layer
        # use 0.04-0.70 m; the real floor is noisier than the simulator's, so cut at the scan band's bottom)
        obst = pts[(pts[:, 2] >= self.scan_band[0]) & (pts[:, 2] <= 1.2)]
        p_l = (obst - T[:3, 3]) @ T[:3, :3]
        return LidarScan(t=self.time, ranges=ranges, angle_min=amin, angle_increment=ainc, range_min=0.35,
                         range_max=6.0, points_lidar=p_l.astype(np.float32), T_world_lidar=T, frame_id="l1_deskewed")

    def camera_pose(self):
        x, y, z, yaw, _ = self._pose()
        c, s = np.cos(yaw), np.sin(yaw)
        Rb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R_cam = Rb @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)    # OpenCV axes, looking forward
        p = np.array([x, y, z - self._floor_z(z)]) + Rb @ self.camera_xyz
        return pose_matrix(p, R_cam)

    def camera(self) -> CameraFrame | None:
        f = self.link.frame()
        if f is None:
            return None
        rgb, _ = f
        h, w = rgb.shape[:2]
        return CameraFrame(t=self.time, rgb=rgb, depth_m=np.full((h, w), np.inf, np.float32),
                           K=intrinsics_from_fovy(w, h, self.camera_fovy_deg), T_world_cam=self.camera_pose(),
                           frame_id="go2_front")

    def depth_points(self, stride=2):
        return np.zeros((0, 3))                    # the Go2's front camera has no depth

    def contacts(self):
        return []                                  # no contact sensing on the real body

    def close(self):
        try:
            self.halt_now()
        finally:
            if self.keys is not None:
                self.keys.close()
            self.link.close()
