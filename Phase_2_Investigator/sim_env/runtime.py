"""In-process simulator session (the "native in-process" route, docs/plan.md section 2).

Wraps the syswonder simulator's own native classes, unmodified:
  NativeSceneBuilder  merges a house environment and the Go2 into one MJCF
  Go2Controller       joint-torque trot gait driven by cmd_vel
  NativeSensorSuite   MID-360 LiDAR (ray cast), IMU, front RGB-D camera
and exposes them through `sim_env.interfaces` types.

    from sim_env.runtime import SimSession
    sim = SimSession("scenesmith_house_185")
    sim.settle()
    sim.cmd_vel(0.3, 0, 0)       # must be re-sent at least every cmd_timeout s
    sim.step(1.0)
    frame, scan, state = sim.camera(), sim.lidar(), sim.state()

Time is simulation time throughout (the gait watchdog uses it too), so a
headless session can run faster or slower than real time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SIM_ROOT = ROOT / "third_party" / "robot-unitree-go2-mujoco"
DEFAULT_ENVIRONMENT = "scenesmith_house_185"

if not os.environ.get("MUJOCO_GL"):
    # Offscreen rendering: GLFW when a display exists (WSLg), else EGL.
    os.environ["MUJOCO_GL"] = "glfw" if os.environ.get("DISPLAY") else "egl"

import mujoco  # noqa: E402

from sim_env.environments import overlay_root  # noqa: E402
from sim_env.scene_xml import base_patch  # noqa: E402
from sim_env.interfaces import (  # noqa: E402
    CameraFrame, Contact, ImuReading, LidarScan, RobotState,
    intrinsics_from_fovy, pose_matrix, quat_to_mat, yaw_from_quat,
)

# MuJoCo camera frame (x right, y up, looking along -z) -> OpenCV (y down, z forward)
_MJ_TO_CV = np.diag([1.0, -1.0, -1.0])
FOOT_GEOMS = ("FL", "FR", "RL", "RR")


def _import_simulator():
    if not (SIM_ROOT / "sim" / "native" / "controller.py").exists():
        raise RuntimeError(
            f"simulator not found at {SIM_ROOT}; run ./scripts/setup.sh "
            "(git submodule update --init --depth 1)")
    if str(SIM_ROOT) not in sys.path:
        sys.path.insert(0, str(SIM_ROOT))
    from sim.native.controller import SCRIPTED_COMMAND_LIMITS, Go2Controller
    from sim.native.scene_builder import NativeSceneBuilder
    from sim.native.sensors import NativeSensorSuite, pinhole_directions
    return NativeSceneBuilder, Go2Controller, NativeSensorSuite, SCRIPTED_COMMAND_LIMITS, pinhole_directions


def list_environments():
    _import_simulator()
    import json
    manifest = json.loads((overlay_root(SIM_ROOT) / "assets/environments/manifest.json").read_text())
    return [e["id"] for e in manifest["environments"] if e.get("visualMode") == "mesh"]


class SimSession:
    """One simulated Go2 in one environment.

    scene_patch: optional callable(scene_xml_path) run on the merged scene file
    before it is compiled; scenarios use it to add stains, mannequins and props
    (A2/A3) without touching the simulator's files.
    spawn: optional (x, y, yaw) overriding the environment's spawn.json.
    """

    def __init__(self, environment=DEFAULT_ENVIRONMENT, scene_patch=None, spawn=None,
                 cmd_timeout=0.5, fix_depth_misses=True):
        Builder, Controller, Sensors, limits, self._pinhole_directions = _import_simulator()
        self.gait_limits = np.array(limits, dtype=float)    # 0.4 m/s, 0.15 m/s, 0.5 rad/s
        self.builder = Builder(overlay_root(SIM_ROOT))
        if spawn is not None:
            env = self.builder.environment(environment)
            env.pop("spawnPath", None)
            env["spawn"] = {"position": [float(spawn[0]), float(spawn[1])],
                            "yaw": float(spawn[2]) if len(spawn) > 2 else 0.0}
        self.scene_path, self.environment, self.robot = self.builder.build(environment, "go2")
        try:
            base_patch(self.scene_path)
            if scene_patch is not None:
                scene_patch(self.scene_path)
            self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
            self.data = mujoco.MjData(self.model)
            self.controller = Controller(self.model, self.data, clock=lambda: self.data.time)
            self.sensors = Sensors(self.model, self.data, self.robot["sensors"])
        except Exception:
            self.builder.cleanup(self.scene_path)
            raise
        self.environment_id = environment
        self.dt = float(self.model.opt.timestep)
        self.cmd_timeout = float(cmd_timeout)
        self.fix_depth_misses = fix_depth_misses
        self._cmd = np.zeros(3)
        self._cmd_time = -np.inf
        self._cmd_resent = -np.inf
        self._renderers = {}
        self._closed = False

        m = self.model
        self.base_body = self.controller.base
        root = int(m.body_rootid[self.base_body])
        self.robot_bodies = {b for b in range(m.nbody) if int(m.body_rootid[b]) == root}
        self.robot_geoms = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.robot_bodies}
        self.foot_geoms = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS}
        mujoco.mj_forward(m, self.data)
        # flat static things the feet walk on (rugs, bath mats, shower trays): count as floor
        self.floorlike_geoms = set()
        for g in range(m.ngeom):
            if g in self.robot_geoms or m.body_mocapid[m.geom_bodyid[g]] >= 0:
                continue
            R = self.data.geom_xmat[g].reshape(3, 3)
            ext = np.abs(R) @ m.geom_size[g] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else None
            if ext is not None and self.data.geom_xpos[g][2] + ext[2] < 0.03:
                self.floorlike_geoms.add(g)
        self.camera_key = next(iter(self.sensors.cameras))
        self.camera_mj_id = self.sensors.camera_ids[self.camera_key]
        self.imu_quat_sensor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ control
    def cmd_vel(self, vx=0.0, vy=0.0, wz=0.0):
        """Body-frame velocity command (ROS Twist linear.x, linear.y, angular.z).

        Like the real robot's dead-man, it expires after cmd_timeout seconds of
        simulation time unless sent again.
        """
        self._cmd[:] = np.clip([vx, vy, wz], -self.gait_limits, self.gait_limits)
        self._cmd_time = self.data.time
        self._send(force=True)

    def stop(self):
        self._cmd[:] = 0
        self._cmd_time = -np.inf
        self.controller.command({"type": "emergency_stop"})

    def reset(self):
        self._cmd[:] = 0
        self._cmd_time = -np.inf
        self.controller.command({"type": "reset"})
        mujoco.mj_forward(self.model, self.data)

    def _send(self, force=False):
        live = self.data.time - self._cmd_time <= self.cmd_timeout
        if live and (force or self.data.time - self._cmd_resent >= 0.1):
            self.controller.command({"type": "cmd_vel", "linearX": float(self._cmd[0]),
                                     "linearY": float(self._cmd[1]), "angularZ": float(self._cmd[2])})
            self._cmd_resent = self.data.time

    @property
    def active_cmd(self):
        live = self.data.time - self._cmd_time <= self.cmd_timeout
        return self._cmd.copy() if live else np.zeros(3)

    def step(self, seconds=None, steps=None):
        """Advance physics by `seconds` (or `steps` physics steps, 2 ms each)."""
        n = steps if steps is not None else max(1, int(round((seconds or self.dt) / self.dt)))
        for _ in range(n):
            self._send()
            self.controller.step()
            mujoco.mj_step(self.model, self.data)

    def settle(self, seconds=1.5):
        """Let the Go2 stand up and come to rest after loading."""
        self.step(seconds)

    @property
    def time(self):
        return float(self.data.time)

    # ------------------------------------------------------------------ sensors
    def state(self) -> RobotState:
        d, c = self.data, self.controller
        q = d.qpos[c.base_qpos:c.base_qpos + 7]
        v = d.qvel[c.base_dof:c.base_dof + 6]
        a = int(self.model.sensor_adr[self.imu_quat_sensor])
        imu = self.sensors.imu()
        return RobotState(
            t=float(d.time), position=q[:3].copy(), quat_wxyz=q[3:7].copy(),
            yaw=yaw_from_quat(q[3:7]), lin_vel_world=v[:3].copy(), ang_vel_body=v[3:6].copy(),
            cmd_vel=self.active_cmd,
            imu=ImuReading(t=imu["timestamp"], gyro=np.array(imu["gyroscope"]),
                           accel=np.array(imu["accelerometer"]),
                           quat_wxyz=d.sensordata[a:a + 4].copy()),
            upright=float(d.xmat[self.base_body, 8]),
        )

    def camera_pose(self):
        """T_world_cam of the front camera in the OpenCV convention."""
        i = self.camera_mj_id
        R = self.data.cam_xmat[i].reshape(3, 3) @ _MJ_TO_CV
        return pose_matrix(self.data.cam_xpos[i].copy(), R)

    def camera(self) -> CameraFrame:
        """Front RGB-D frame; depth (ray cast at 160x120) is upsampled to the RGB size."""
        f = self.sensors.camera(self.camera_key)
        rgb = np.ascontiguousarray(f["rgb"][:, :, :3])
        H, W = rgb.shape[:2]
        dd = f["depth"]
        depth = dd["data"].reshape(dd["height"], dd["width"])
        if self.fix_depth_misses:
            depth = self._recast_depth_misses(depth, f["fovyDeg"])
        rows = (np.arange(H) * dd["height"]) // H
        cols = (np.arange(W) * dd["width"]) // W
        depth = depth[rows][:, cols]
        return CameraFrame(t=f["timestamp"], rgb=rgb, depth_m=depth,
                           K=intrinsics_from_fovy(W, H, f["fovyDeg"]),
                           T_world_cam=self.camera_pose(), frame_id=self.camera_key)

    def depth_points(self, stride=2):
        """World points of the front depth camera (ray cast only, no RGB render; cheap).

        Used by navigation to see low obstacles the LiDAR misses up close."""
        cfg = self.sensors.cameras[self.camera_key]
        i = self.camera_mj_id
        w, h = int(cfg.get("depthWidth", cfg["width"])), int(cfg.get("depthHeight", cfg["height"]))
        key = ("dirs", w, h, stride)
        if key not in self._renderers:
            dirs = self._pinhole_directions(w, h, float(self.model.cam_fovy[i])).reshape(h, w, 3)
            self._renderers[key] = np.ascontiguousarray(dirs[::stride, ::stride].reshape(-1, 3))
        local = self._renderers[key]
        R = self.data.cam_xmat[i].reshape(3, 3)
        world = local @ R.T
        n = len(local)
        dist = np.empty(n)
        geom = np.empty(n, np.int32)
        mujoco.mj_multiRay(self.model, self.data, self.data.cam_xpos[i], world.reshape(-1), self.sensors.geom_group,
                           1, int(self.model.cam_bodyid[i]), geom, dist, n, float(cfg["maxDepth"]))
        ok = (dist > 0) & (dist * -local[:, 2] <= float(cfg["maxDepth"]))
        return self.data.cam_xpos[i] + world[ok] * dist[ok, None]

    def _recast_depth_misses(self, depth, fovy_deg):
        """Re-cast missing depth pixels one ray at a time.

        MuJoCo 3.3.7's batched mj_multiRay (used by the simulator's camera)
        misses some box geoms that mj_ray hits, e.g. a few % of the floor near
        doorways in house 186. Only the missing pixels are re-cast, so this is
        cheap; true misses (open sky, beyond max depth) stay inf.
        """
        bad = np.flatnonzero(~np.isfinite(depth))
        if bad.size == 0:
            return depth
        h, w = depth.shape
        cfg = self.sensors.cameras[self.camera_key]
        i = self.camera_mj_id
        local = self._pinhole_directions(w, h, fovy_deg)[bad]
        world = local @ self.data.cam_xmat[i].reshape(3, 3).T
        geom = np.zeros(1, np.int32)
        body = int(self.model.cam_bodyid[i])
        out = depth.copy().reshape(-1)
        for k, idx in enumerate(bad):
            dist = mujoco.mj_ray(self.model, self.data, self.data.cam_xpos[i], world[k],
                                 self.sensors.geom_group, 1, body, geom)
            z = dist * -local[k, 2]
            if dist >= 0 and cfg["minDepth"] <= z <= cfg["maxDepth"]:
                out[idx] = z
        return out.reshape(h, w)

    def lidar(self) -> LidarScan:
        scan, cloud = self.sensors.lidar()
        s = self.sensors.lidar_site
        T = pose_matrix(self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy())
        return LidarScan(t=scan["timestamp"], ranges=scan["ranges"], angle_min=scan["angleMin"],
                         angle_increment=scan["angleIncrement"], range_min=scan["rangeMin"],
                         range_max=scan["rangeMax"], points_lidar=cloud["points"], T_world_lidar=T)

    def contacts(self):
        """Robot contacts other than feet on the ground: collisions to log.

        A contact counts when a non-foot robot geom touches anything outside the
        robot, or a foot touches something on a steep face (furniture, walls).
        Feet on flat floor coverings (boxes under 3 cm: rugs, mats) never count.
        """
        m, d = self.model, self.data
        out = []
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            r1, r2 = g1 in self.robot_geoms, g2 in self.robot_geoms
            if r1 == r2:
                continue
            rg, og = (g1, g2) if r1 else (g2, g1)
            normal = np.array(con.frame[:3])
            if rg in self.foot_geoms and (abs(normal[2]) > 0.7 or og in self.floorlike_geoms):
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            out.append(Contact(
                t=float(d.time), robot_geom=_name(m, mujoco.mjtObj.mjOBJ_GEOM, rg),
                other_geom=_name(m, mujoco.mjtObj.mjOBJ_GEOM, og),
                other_body=_name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[og])),
                position=np.array(con.pos), normal=normal, force=float(abs(f6[0]))))
        return out

    # ---------------------------------------------------------------- rendering
    def render_view(self, width=640, height=480, distance=1.6, azimuth=None, elevation=-35.0,
                    lookat=None, camera=None):
        """Third-person view (chase camera behind the robot by default)."""
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            r = self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        if camera is None:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            st = self.state()
            camera.lookat[:] = st.position if lookat is None else lookat
            camera.distance = distance
            camera.azimuth = np.degrees(st.yaw) if azimuth is None else azimuth
            camera.elevation = elevation
        r.update_scene(self.data, camera=camera, scene_option=self.sensors.render_option)
        return r.render().copy()

    def render_camera(self, position, forward, fovy=40.0, width=1920, height=1080, depth=True,
                      frame_id="evidence", segmentation=False):
        """Render a pinhole camera at `position` looking along `forward` (no roll).

        Used for evidence close-ups: the robot's camera position with an emulated
        body pitch, at a higher resolution than the 320x240 sensor (see docs/plan.md).
        Depth is the rendered z-depth (includes the stains, which sit 0.8 mm proud).
        segmentation=True also returns a (H, W) array of geom ids (-1 = background).
        """
        f = np.asarray(forward, float)
        f /= np.linalg.norm(f)
        key = (width, height)
        r = self._renderers.get(key)
        if r is None:
            r = self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = np.asarray(position, float) + f
        cam.distance = 1.0
        cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1, 1))))
        old = float(self.model.vis.global_.fovy)
        self.model.vis.global_.fovy = fovy
        try:
            r.update_scene(self.data, camera=cam, scene_option=self.sensors.render_option)
            rgb = r.render().copy()
            if depth:
                r.enable_depth_rendering()
                r.update_scene(self.data, camera=cam, scene_option=self.sensors.render_option)
                d = r.render().astype(np.float32)
                r.disable_depth_rendering()
                d[d >= self.model.stat.extent * self.model.vis.map.zfar * 0.99] = np.inf
            else:
                d = np.full(rgb.shape[:2], np.inf, np.float32)
            seg = None
            if segmentation:
                r.enable_segmentation_rendering()
                r.update_scene(self.data, camera=cam, scene_option=self.sensors.render_option)
                s = r.render()
                r.disable_segmentation_rendering()
                seg = np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1).astype(np.int32)
        finally:
            self.model.vis.global_.fovy = old
        x = np.cross(f, [0.0, 0.0, 1.0])
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9 else np.array([0.0, -1.0, 0.0])
        y = np.cross(f, x)
        T = pose_matrix(np.asarray(position, float), np.column_stack([x, y, f]))
        frame = CameraFrame(t=self.time, rgb=rgb, depth_m=d, K=intrinsics_from_fovy(width, height, fovy),
                            T_world_cam=T, frame_id=frame_id)
        return (frame, seg) if segmentation else frame

    def evidence_frame(self, target, fovy=40.0, width=1920, height=1080, max_pitch_deg=40.0):
        """Close-up from the robot's camera position toward a world point (pitch clamped)."""
        cam = self.camera_pose()[:3, 3]
        d = np.asarray(target, float) - cam
        yaw = np.arctan2(d[1], d[0])
        pitch = np.clip(np.arctan2(d[2], np.hypot(d[0], d[1])), -np.radians(max_pitch_deg),
                        np.radians(max_pitch_deg))
        fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        return self.render_camera(cam, fwd, fovy, width, height)

    # -------------------------------------------------------------------- close
    def close(self):
        if self._closed:
            return
        self._closed = True
        self.sensors.close()
        for r in self._renderers.values():
            if hasattr(r, "close"):
                r.close()
        self._renderers.clear()
        self.builder.cleanup(self.scene_path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _name(model, kind, idx):
    name = mujoco.mj_id2name(model, kind, idx)
    if name:
        return name
    if kind == mujoco.mjtObj.mjOBJ_GEOM:   # unnamed geom: name it after its body
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[idx]))
        return f"{body}/geom{idx}"
    return f"{kind.name.split('_')[-1].lower()}{idx}"


def quat_from_yaw(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


__all__ = ["SimSession", "list_environments", "SIM_ROOT", "DEFAULT_ENVIRONMENT", "quat_to_mat",
           "quat_from_yaw"]
