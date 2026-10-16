"""MuJoCo simulation of the Go2 in an indoor room, for developing perception without the robot.

Unitree's own simulator (unitree_mujoco) only accepts low-level joint commands over DDS: it has no
SportClient service (no Move/StandUp), no camera and no LiDAR. So this module runs the Go2 model in a
KINEMATIC mode: the base is moved directly from the commanded velocity, and the legs blend between a
lying and a standing pose. It is meant for perception/mapping tests, not for testing gaits.

SimWorld exposes the same surface as camera.CameraThread (.buffer, .fps, .start(), .stop()), so the
detector cannot tell sim frames from real ones. SimSportClient has the same methods as SportClient,
so teleop.Teleop drives the sim unchanged.
"""
import logging
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml

from camera import LatestFrame
from perception.depth_fusion import CameraModel

log = logging.getLogger("sim")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# keyframe "home" of go2.xml, and a lying-down pose
HOME_JOINTS = np.array([0, 0.9, -1.8] * 4, dtype=float)
LIE_JOINTS = np.array([0, 1.36, -2.65] * 4, dtype=float)
STAND_Z, LIE_Z = 0.27, 0.09
STAND_TIME_S = 1.5


@dataclass
class SimObject:
    name: str            # body name, e.g. obj_chair_1
    cls: str             # "chair", "dining table", ...
    center: np.ndarray   # true world position (centre of the object's bounding box), metres
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    corners: np.ndarray  # (N,3) world-frame points of all geom bounding boxes (for projection)


@dataclass
class SimBundle:
    """Everything that belongs to one rendered frame."""
    seq: int
    bgr: np.ndarray
    depth: np.ndarray        # float32 metres along the camera's optical axis; <=0 or inf = no return
    cam_pos: np.ndarray      # world position of the camera
    cam_mat: np.ndarray      # 3x3, columns = camera x(right), y(up), z(backward) in world (MuJoCo convention)
    robot_xyyaw: tuple
    t: float


# ---------------------------------------------------------------------------------------------- scene
def load_config(path=None):
    path = path or os.path.join(ROOT, "config", "sim.yaml")
    cfg = yaml.safe_load(open(path))
    cfg["_path"] = path
    return cfg


def unitree_mujoco_dir(cfg):
    d = os.path.expanduser(os.environ.get("UNITREE_MUJOCO") or cfg["unitree_mujoco_dir"])
    xml = os.path.join(d, cfg["robot_xml"])
    if not os.path.isfile(xml):
        raise FileNotFoundError(
            f"Go2 model not found at {xml}.\nInstall Unitree's simulator model with:\n"
            f"  git clone https://github.com/unitreerobotics/unitree_mujoco.git {d}\n"
            f"(or set UNITREE_MUJOCO=/path/to/unitree_mujoco)")
    return d, xml


def _billboards_xml(cfg):
    assets, bodies = [], []
    for b in cfg.get("billboards") or []:
        img = os.path.join(ROOT, "sim", "photos", b["image"])
        if not os.path.isfile(img):
            log.warning("billboard image %s not found, skipping %s (run python sim/fetch_photos.py)", img, b["name"])
            continue
        n = b["name"]
        w, h = b["size"]
        assets.append(f'<texture name="tex_{n}" type="2d" file="{img}"/>'
                      f'<material name="mat_{n}" texture="tex_{n}" texuniform="false"/>')
        yaw = math.radians(b.get("yaw_deg", 0.0))
        x, y, z = b["pos"]
        # thin board: thickness along x, width along y, height along z (faces +x at yaw 0)
        bodies.append(f'<body name="{n}" pos="{x} {y} {z}" euler="0 0 {yaw:.5f}">'
                      f'<geom type="box" size="0.01 {w / 2} {h / 2}" material="mat_{n}"/></body>')
    return "\n".join(assets), "\n".join(bodies)


def build_model(cfg):
    """Compile scene_room.xml (+ billboards) with the Unitree Go2 model and a head camera.

    Returns (model, data, objects, info) where info has camera/robot ids.
    """
    import mujoco
    ud, go2_xml = unitree_mujoco_dir(cfg)
    template = open(os.path.join(ROOT, cfg["scene"])).read()
    bb_assets, bb_bodies = _billboards_xml(cfg)
    xml = (template.replace("{{GO2_XML}}", "go2.xml").replace("{{BILLBOARD_ASSETS}}", bb_assets)
           .replace("{{BILLBOARDS}}", bb_bodies))
    build_dir = os.path.join(HERE, "_build")
    os.makedirs(build_dir, exist_ok=True)
    # MuJoCo resolves mesh paths relative to the main model's directory, so link the Unitree model into the
    # build dir instead of copying or editing the third-party clone.
    for name, target in (("go2.xml", go2_xml), ("assets", os.path.join(os.path.dirname(go2_xml), "assets"))):
        link = os.path.join(build_dir, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)
    scene_path = os.path.join(build_dir, "scene_built.xml")
    open(scene_path, "w").write(xml)

    spec = mujoco.MjSpec.from_file(scene_path)
    cam = cfg["camera"]
    pitch = math.radians(cam.get("pitch_deg", 0.0))
    # MuJoCo camera looks along its -z. Forward = body +x, right = body -y, up = body +z; then pitch down.
    c, s = math.cos(pitch), math.sin(pitch)
    zx = [0, -1, 0]                                   # camera x (right) in the body frame
    yv = [-s, 0, c]                                   # camera y (up) in the body frame, pitched down by `pitch`
    spec.body("base_link").add_camera(name="front", pos=cam["position"], xyaxes=zx + yv)
    model = spec.compile()
    data = mujoco.MjData(model)
    # fovy from the horizontal FOV so the sim and detection.yaml use the same camera model
    w, h = cam["width"], cam["height"]
    model.cam_fovy[model.cam("front").id] = math.degrees(2 * math.atan(math.tan(math.radians(cam["hfov_deg"]) / 2) * h / w))
    base = model.body("base_link").id
    for g in range(model.ngeom):
        if model.body_rootid[model.geom_bodyid[g]] == base:
            model.geom_group[g] = 5          # hidden from the default render and ignored by our ray casts
    objects = _collect_objects(model, data)
    return model, data, objects


def _geom_corners(model, data, gid):
    """8 world-frame corners of a geom's local bounding box (exact for boxes, bounding for cylinders)."""
    sx, sy, sz = model.geom_size[gid]
    gtype = model.geom_type[gid]
    import mujoco
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        ext = np.array([sx, sx, sz])
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        ext = np.array([sx, sx, sx])
    else:
        ext = np.array([sx, sy, sz])
    signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    local = signs * ext
    return data.geom_xpos[gid] + local @ data.geom_xmat[gid].reshape(3, 3).T


def _collect_objects(model, data) -> List[SimObject]:
    import mujoco
    mujoco.mj_forward(model, data)
    out = []
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        m = re.match(r"obj_(.+)_(\d+)$", name)
        if not m:
            continue
        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
        if not gids:
            continue
        corners = np.vstack([_geom_corners(model, data, g) for g in gids])
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        out.append(SimObject(name, m.group(1).replace("_", " "), (lo + hi) / 2, lo, hi, corners))
    return out


# ----------------------------------------------------------------------------------------------- world
class SimWorld(threading.Thread):
    """Runs the kinematic sim and renders the front camera. Thread-safe command interface."""

    def __init__(self, cfg=None, render=None):
        super().__init__(name="sim", daemon=True)
        import mujoco
        self.mj = mujoco
        self.cfg = cfg or load_config()
        self.model, self.data, self.objects = build_model(self.cfg)
        cam = self.cfg["camera"]
        self.width, self.height, self.fps_target = cam["width"], cam["height"], cam["fps"]
        self.camera_model = CameraModel.from_hfov(self.width, self.height, cam["hfov_deg"])
        self.cam_id = self.model.cam("front").id
        self.base_id = self.model.body("base_link").id
        self.floor_gid = self.model.geom("floor").id
        self.render_enabled = self.cfg.get("render", True) if render is None else render
        self.render_backend = "none"

        sp = self.cfg["start_pose"]
        self._x, self._y, self._yaw = sp["x"], sp["y"], math.radians(sp["yaw_deg"])
        self._stand_p = 1.0 if sp.get("standing") else 0.0
        self._stand_target = self._stand_p
        self._stand_rate = 1.0 / STAND_TIME_S

        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)
        self._cmd_t = 0.0
        self._stop_evt = threading.Event()
        self.buffer = LatestFrame()      # same API/semantics as CameraThread.buffer (BGR frames)
        self.fps = 0.0
        self._seq = 0
        self._bundles = deque(maxlen=8)
        self.trail = [(self._x, self._y)]
        self._apply_pose()

    # ---- command interface (called from teleop / SimSportClient threads)
    def set_velocity(self, vx, vy, vyaw):
        with self._lock:
            if self._stand_p < 1.0 or self._stand_target < 1.0:
                return 1                      # like the real robot, Move is refused unless standing
            self._cmd, self._cmd_t = (vx, vy, vyaw), time.monotonic()
        return 0

    def stop_move(self):
        with self._lock:
            self._cmd = (0.0, 0.0, 0.0)
        return 0

    def stand(self, up, fast=False):
        with self._lock:
            self._stand_target = 1.0 if up else 0.0
            self._stand_rate = (3.0 if fast else 1.0) / STAND_TIME_S
            if not up:
                self._cmd = (0.0, 0.0, 0.0)
        return 0

    @property
    def pose(self):
        with self._lock:
            return self._x, self._y, self._yaw

    @property
    def standing(self):
        return self._stand_p >= 1.0

    def stop(self):
        self._stop_evt.set()

    # ---- state
    def bundle_for(self, frame) -> Optional[SimBundle]:
        for b in reversed(self._bundles):
            if b.bgr is frame:
                return b
        return None

    def latest_bundle(self) -> Optional[SimBundle]:
        return self._bundles[-1] if self._bundles else None

    def _apply_pose(self):
        d, mj = self.data, self.mj
        p = self._stand_p
        z = LIE_Z + (STAND_Z - LIE_Z) * p
        d.qpos[0:3] = [self._x, self._y, z]
        d.qpos[3:7] = [math.cos(self._yaw / 2), 0, 0, math.sin(self._yaw / 2)]
        d.qpos[7:19] = LIE_JOINTS + (HOME_JOINTS - LIE_JOINTS) * p
        d.qvel[:] = 0
        mj.mj_forward(self.model, d)

    def _robot_collides(self):
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist >= 0:
                continue
            r1 = m.body_rootid[m.geom_bodyid[c.geom1]] == self.base_id
            r2 = m.body_rootid[m.geom_bodyid[c.geom2]] == self.base_id
            if r1 == r2:
                continue
            other = c.geom2 if r1 else c.geom1
            if other != self.floor_gid:
                return True
        return False

    def _step(self, dt):
        with self._lock:
            cmd = self._cmd
            if time.monotonic() - self._cmd_t > self.cfg["command_timeout_s"]:
                cmd = (0.0, 0.0, 0.0)
            tgt, rate = self._stand_target, self._stand_rate
        stand_changed = self._stand_p != tgt
        if stand_changed:
            step = rate * dt
            self._stand_p = tgt if abs(tgt - self._stand_p) <= step else self._stand_p + math.copysign(step, tgt - self._stand_p)
        vx, vy, wz = cmd
        old = (self._x, self._y, self._yaw)
        if vx or vy or wz:
            c, s = math.cos(self._yaw), math.sin(self._yaw)
            new = (self._x + (vx * c - vy * s) * dt, self._y + (vx * s + vy * c) * dt, self._yaw + wz * dt)
            with self._lock:
                self._x, self._y, self._yaw = new
            self._apply_pose()
            if self._robot_collides():       # blocked by furniture/wall: undo this step
                with self._lock:
                    self._x, self._y, self._yaw = old
                self._apply_pose()
            elif not self.trail or math.hypot(self._x - self.trail[-1][0], self._y - self.trail[-1][1]) > 0.05:
                self.trail.append((self._x, self._y))
        elif stand_changed:
            self._apply_pose()

    # ---- camera
    def _cam_pose(self):
        return self.data.cam_xpos[self.cam_id].copy(), self.data.cam_xmat[self.cam_id].reshape(3, 3).copy()

    def _make_renderer(self):
        if not self.render_enabled:
            return None
        try:
            r = self.mj.Renderer(self.model, self.height, self.width)
            self.render_backend = os.environ.get("MUJOCO_GL", "glfw")
            return r
        except Exception as e:
            log.error("MuJoCo offscreen rendering failed (%s). Falling back to a flat-colour ray-cast view; "
                      "YOLO will not recognise it. Try MUJOCO_GL=egl (NVIDIA) or osmesa.", e)
            return None

    def _raycast_frame(self):
        """No-OpenGL fallback: ray-cast depth + flat colours. Slow but dependable; used for headless tests."""
        mj, m, d = self.mj, self.model, self.data
        pos, mat = self._cam_pose()
        cm = self.camera_model
        us, vs = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        x = (us - cm.cx) / cm.fx
        y = (vs - cm.cy) / cm.fy
        dirs_opt = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
        dirs_mj = np.stack([dirs_opt[:, 0], -dirs_opt[:, 1], -dirs_opt[:, 2]], axis=1)
        vec = (dirs_mj @ mat.T)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        n = len(vec)
        geomid = np.full(n, -1, dtype=np.int32)
        dist = np.full(n, -1.0)
        flat = np.ascontiguousarray(vec.reshape(-1))
        groups = np.array([1, 1, 1, 1, 1, 0], dtype=np.uint8)
        try:
            mj.mj_multiRay(m, d, pos, flat, groups, 1, -1, geomid, dist, None, n, 50.0)
        except TypeError:                                   # older MuJoCo without the `normal` argument
            mj.mj_multiRay(m, d, pos, flat, groups, 1, -1, geomid, dist, n, 50.0)
        hit = geomid >= 0
        zdepth = np.where(hit, dist * (dirs_opt[:, 2] / np.linalg.norm(dirs_opt, axis=1)), 0.0)
        rgba = np.where(hit[:, None], m.geom_rgba[np.maximum(geomid, 0)][:, :3], 0.0)
        mat_ids = m.geom_matid[np.maximum(geomid, 0)]
        has_mat = hit & (mat_ids >= 0)
        rgba[has_mat] = m.mat_rgba[mat_ids[has_mat]][:, :3]
        bgr = (rgba[:, ::-1] * 255).astype(np.uint8).reshape(self.height, self.width, 3)
        return bgr, zdepth.astype(np.float32).reshape(self.height, self.width)

    def _render(self, renderer):
        self.mj.mj_forward(self.model, self.data)
        if renderer is None:
            bgr, depth = self._raycast_frame()
        else:
            renderer.update_scene(self.data, camera="front")
            rgb = renderer.render()
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            renderer.enable_depth_rendering()
            renderer.update_scene(self.data, camera="front")
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
        pos, mat = self._cam_pose()
        self._seq += 1
        b = SimBundle(self._seq, bgr, depth, pos, mat, (self._x, self._y, self._yaw), time.time())
        self._bundles.append(b)
        self.buffer.put(bgr)
        return b

    # ---- thread
    def run(self):
        period = 1.0 / self.cfg["physics_hz"]
        renderer = self._make_renderer()
        if renderer is None:
            self.render_backend = "raycast"
        log.info("Sim running: render=%s, camera %dx%d hfov %.0f deg, %d labelled objects",
                 self.render_backend, self.width, self.height, self.cfg["camera"]["hfov_deg"], len(self.objects))
        next_t, next_cam, times = time.monotonic(), 0.0, []
        last = next_t
        while not self._stop_evt.is_set():
            try:
                now = time.monotonic()
                self._step(min(now - last, 0.1))     # real elapsed time, so rendering delays don't slow the robot
                last = now
                if now >= next_cam:
                    self._render(renderer)
                    next_cam = now + 1.0 / self.fps_target
                    times.append(now)
                    while times and now - times[0] > 2.0:
                        times.pop(0)
                    self.fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
            except Exception:
                log.exception("sim step failed")
                time.sleep(0.2)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
        if renderer is not None:
            renderer.close()


class SimSportClient:
    """Same methods as unitree_sdk2py's SportClient, backed by SimWorld, so teleop.Teleop works unchanged."""

    def __init__(self, world: SimWorld):
        self.world = world

    def Move(self, vx, vy, vyaw): return self.world.set_velocity(vx, vy, vyaw)
    def StopMove(self): return self.world.stop_move()
    def StandUp(self): return self.world.stand(True)
    def BalanceStand(self): return 0
    def StandDown(self): return self.world.stand(False)
    def Damp(self): return self.world.stand(False, fast=True)
