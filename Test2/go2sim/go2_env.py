"""The Go2 in the crime scene: one clean interface for Q2 data collection and Q3 deployment.

    from go2sim.go2_env import CrimeSceneGo2
    dog = CrimeSceneGo2("1")                    # Phase 2 scenario S1 (house 185), learned policy
    dog.set_velocity_command(0.3, 0.0, 0.2)     # body frame: vx (m/s), vy (m/s), yaw rate (rad/s)
    dog.step(0.5)                               # advance physics 0.5 s (command re-sent every 20 ms)
    s = dog.read_state()                        # flat dict: IMU, joints, feet, base, command
    rgb = dog.render_front()                    # uint8 HxWx3 from the head camera

What it reuses (imported unchanged from ../Phase_2_Investigator):
- `open_scenario`: builds the scenario (house, mannequins, props, distractors, bloodstains) and
  returns a `SimSession` plus the ground truth (stain clusters = evidence);
- `SimSession`: MuJoCo model/data, Go2Controller (simulator's), sensors, contacts.

Locomotion: the simulator's Go2Controller supports two controllers, both driving the 12 joint
torque motors of the official unitree_mujoco go2.xml through real contact physics:
- "learned" (default): the released "MoE Rough" Go2 RL locomotion policy (ONNX, 45-float
  observation x 5-frame history -> 12 joint-position targets at 50 Hz, PD kp=20 kd=0.5 -> torque).
  unitree_rl_gym ships no pretrained Go2 policy (only G1/H1/H1_2), so this is the pretrained Go2
  velocity-tracking policy we use instead.
- "scripted": the simulator's analytic diagonal trot with joint PD + support torques (fallback).
Commands are clipped to the gait limits 0.4 m/s forward, 0.15 m/s sideways, 0.5 rad/s.
"""
from __future__ import annotations

import numpy as np

from go2sim import paths  # noqa: F401  (sys.path + GL backend before mujoco)

import mujoco  # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
JOINTS = [f"{leg}_{j}" for leg in LEGS for j in ("hip", "thigh", "calf")]
# Three ceiling lamps added to the scene (dark by default) so lighting can be randomised per run;
# the house itself is lit only by MuJoCo's headlight.
LAMPS = [(1.2, 2.2, 2.4), (3.6, 2.2, 2.4), (7.1, 2.25, 2.3)]
LAMP_XML = "".join(f'<light name="t2_lamp_{i}" pos="{x} {y} {z}" dir="0 0 -1" directional="false" '
                   f'diffuse="0 0 0" specular="0 0 0" attenuation="1 0.1 0.05" castshadow="false"/>'
                   for i, (x, y, z) in enumerate(LAMPS))
CONTROL_DT = 0.02          # command re-send / policy period (s)
FOOT_CONTACT_N = 5.0       # a foot is "in contact" above this normal force (N)


def _rot_body_to_world(quat_wxyz):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat_wxyz, float))
    return m.reshape(3, 3)


class CrimeSceneGo2:
    """One Go2 in one crime-scene scenario."""

    def __init__(self, scenario="1", spawn=None, controller="learned", camera_size=(320, 240),
                 show_stains=True, settle_s=1.5):
        from sim_env.scenario_scene import load_scenario_config, open_scenario
        self.scenario = scenario if isinstance(scenario, dict) else load_scenario_config(str(scenario))
        from sim_env.scene_xml import append_xml
        self.sim, self.gt = open_scenario(self.scenario, show_stains=show_stains, spawn=spawn,
                                          extra=lambda root: append_xml(root, "worldbody", LAMP_XML))
        self.model, self.data = self.sim.model, self.sim.data
        c = self.ctrl = self.sim.controller
        self.controller_name = controller
        self.sim.settle(settle_s)
        if controller == "learned":
            if not c.command({"type": "policy", "action": "load"}) or not c.policy_status["loaded"]:
                raise RuntimeError(f"learned policy failed to load: {c.policy_status}")
            self.sim.step(1.0)                       # blend into the policy's stance
        elif controller != "scripted":
            raise ValueError("controller must be 'learned' or 'scripted'")
        self.limits = self.sim.gait_limits.copy()   # [0.4, 0.15, 0.5]
        self._cmd = np.zeros(3)
        m = self.model
        self.foot_geom = {leg: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in LEGS}
        self._renderers = {}
        self._light0 = (m.light_diffuse.copy(), m.light_ambient.copy(), m.light_specular.copy(),
                        m.light_pos.copy(), m.light_dir.copy(), m.vis.headlight.ambient.copy(),
                        m.vis.headlight.diffuse.copy())
        cam = self.sim.camera_mj_id
        self.camera_fovy_deg = float(m.cam_fovy[cam])
        self.camera_mount = {"body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.cam_bodyid[cam])),
                             "pos_in_base_m": m.cam_pos[cam].round(4).tolist(),
                             "fovy_deg": self.camera_fovy_deg}
        self.camera_size = tuple(camera_size)

    # ------------------------------------------------------------------ control
    def set_velocity_command(self, vx: float, vy: float, yaw_rate: float):
        """Body-frame velocity command, clipped to the Go2 gait limits. Held until changed."""
        self._cmd = np.clip([vx, vy, yaw_rate], -self.limits, self.limits).astype(float)
        self.sim.cmd_vel(*self._cmd)

    def stop(self):
        self.set_velocity_command(0.0, 0.0, 0.0)

    def teleport(self, x, y, yaw, settle_s=1.0):
        """Place the standing dog at (x, y, yaw) with zero velocity, then let it settle.

        Lets one loaded scene serve several runs with different start poses."""
        c, d = self.ctrl, self.data
        q = d.qpos[c.base_qpos:c.base_qpos + 7]
        q[0], q[1] = x, y
        q[2] = max(q[2], 0.33)
        q[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        d.qvel[c.base_dof:c.base_dof + 6] = 0.0
        d.qvel[c.dof] = 0.0
        mujoco.mj_forward(self.model, d)
        self._cmd[:] = 0
        self.step(settle_s)
        return self.pose2d()

    @property
    def command(self):
        return self._cmd.copy()

    def step(self, seconds=CONTROL_DT):
        """Advance physics by `seconds`, re-sending the held command every CONTROL_DT."""
        n = max(1, int(round(seconds / CONTROL_DT)))
        for _ in range(n):
            self.sim.cmd_vel(*self._cmd)
            self.sim.step(CONTROL_DT)

    @property
    def time(self):
        return float(self.data.time)

    # ------------------------------------------------------------------ state
    def pose2d(self):
        st = self.sim.state()
        return np.array([st.position[0], st.position[1], st.yaw])

    def foot_forces(self):
        """Normal contact force (N) on each foot from anything outside the robot."""
        out = dict.fromkeys(LEGS, 0.0)
        gid = {g: leg for leg, g in self.foot_geom.items()}
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            for g, other in ((g1, g2), (g2, g1)):
                if g in gid and other not in self.sim.robot_geoms:
                    mujoco.mj_contactForce(self.model, self.data, i, f6)
                    out[gid[g]] += abs(f6[0])
        return out

    def read_state(self) -> dict:
        """Everything the dog senses, as one flat row (SI units, world frame unless *_body)."""
        st = self.sim.state()
        c, d = self.ctrl, self.data
        R = _rot_body_to_world(st.quat_wxyz)
        v_body = R.T @ st.lin_vel_world
        row = {"t": round(float(d.time), 4),
               "base_x": st.position[0], "base_y": st.position[1], "base_z": st.position[2],
               "base_qw": st.quat_wxyz[0], "base_qx": st.quat_wxyz[1], "base_qy": st.quat_wxyz[2],
               "base_qz": st.quat_wxyz[3], "base_yaw": st.yaw,
               "base_vx_world": st.lin_vel_world[0], "base_vy_world": st.lin_vel_world[1],
               "base_vz_world": st.lin_vel_world[2],
               "base_vx_body": v_body[0], "base_vy_body": v_body[1], "base_vz_body": v_body[2],
               "base_wx_body": st.ang_vel_body[0], "base_wy_body": st.ang_vel_body[1],
               "base_wz_body": st.ang_vel_body[2],
               "imu_qw": st.imu.quat_wxyz[0], "imu_qx": st.imu.quat_wxyz[1],
               "imu_qy": st.imu.quat_wxyz[2], "imu_qz": st.imu.quat_wxyz[3],
               "imu_gyro_x": st.imu.gyro[0], "imu_gyro_y": st.imu.gyro[1], "imu_gyro_z": st.imu.gyro[2],
               "imu_acc_x": st.imu.accel[0], "imu_acc_y": st.imu.accel[1], "imu_acc_z": st.imu.accel[2],
               "upright": st.upright,
               "cmd_vx": self._cmd[0], "cmd_vy": self._cmd[1], "cmd_yaw_rate": self._cmd[2]}
        q = d.qpos[c.qpos]
        dq = d.qvel[c.dof]
        tau = d.actuator_force[c.actuators]
        for i, j in enumerate(JOINTS):
            row[f"q_{j}"] = q[i]
        for i, j in enumerate(JOINTS):
            row[f"dq_{j}"] = dq[i]
        for i, j in enumerate(JOINTS):
            row[f"tau_{j}"] = tau[i]
        ff = self.foot_forces()
        for leg in LEGS:
            row[f"foot_force_{leg}"] = ff[leg]
        for leg in LEGS:
            row[f"foot_contact_{leg}"] = int(ff[leg] > FOOT_CONTACT_N)
        return {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}

    def collisions(self):
        """Robot contacts other than feet on the floor (furniture, walls, props, mannequins)."""
        return self.sim.contacts()

    # ------------------------------------------------------------------ rendering
    def _renderer(self, w, h):
        r = self._renderers.get((w, h))
        if r is None:
            r = self._renderers[(w, h)] = mujoco.Renderer(self.model, height=h, width=w)
        return r

    def render_front(self, size=None):
        """RGB frame from the Go2's head camera (the simulator's front_rgbd_camera)."""
        w, h = size or self.camera_size
        r = self._renderer(w, h)
        r.update_scene(self.data, camera=self.sim.camera_mj_id, scene_option=self.sim.sensors.render_option)
        return r.render().copy()

    def render_chase(self, size=(480, 360), distance=2.0, elevation=-25.0):
        w, h = size
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        st = self.sim.state()
        cam.lookat[:] = st.position
        cam.distance, cam.azimuth, cam.elevation = distance, np.degrees(st.yaw), elevation
        r = self._renderer(w, h)
        r.update_scene(self.data, camera=cam, scene_option=self.sim.sensors.render_option)
        return r.render().copy()

    def render_topdown(self, extent=(-0.5, 8.7, -0.5, 5.0), px_per_m=100):
        """Orthographic top view covering `extent` (xmin, xmax, ymin, ymax) exactly, for map plots."""
        x0, x1, y0, y1 = extent
        w, h = int(round((x1 - x0) * px_per_m)), int(round((y1 - y0) * px_per_m))
        m = self.model
        fovy0 = float(m.vis.global_.fovy)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.orthographic = 1
        cam.lookat[:] = [(x0 + x1) / 2, (y0 + y1) / 2, 0.0]
        cam.distance, cam.azimuth, cam.elevation = 10.0, 90.0, -90.0
        m.vis.global_.fovy = y1 - y0               # orthographic: vertical extent in metres
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.geomgroup[:] = self.sim.sensors.render_option.geomgroup
        try:
            r = self._renderer(w, h) if w <= m.vis.global_.offwidth and h <= m.vis.global_.offheight \
                else self._renderer(min(w, m.vis.global_.offwidth), min(h, m.vis.global_.offheight))
            r.update_scene(self.data, camera=cam, scene_option=opt)
            return r.render().copy()
        finally:
            m.vis.global_.fovy = fovy0

    # ------------------------------------------------------------------ randomisation
    def set_lighting(self, headlight_ambient=1.0, headlight_diffuse=1.0, tint=(1, 1, 1),
                     lamps=(0.0, 0.0, 0.0), lamp_shift=(0.0, 0.0)):
        """Randomisable lighting: headlight scales (x the house default), a colour tint, the
        brightness (0-1) of each added ceiling lamp, and a horizontal shift of the lamps (m)."""
        dif, amb, spec, pos, dirn, hamb, hdif = self._light0
        m = self.model
        tint = np.asarray(tint, float)
        m.vis.headlight.ambient[:] = np.clip(hamb * headlight_ambient * tint, 0, 1)
        m.vis.headlight.diffuse[:] = np.clip(hdif * headlight_diffuse * tint, 0, 1)
        for i in range(len(LAMPS)):
            lid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, f"t2_lamp_{i}")
            m.light_diffuse[lid] = np.clip(float(lamps[i]) * tint, 0, 1)
            m.light_pos[lid] = pos[lid] + [lamp_shift[0], lamp_shift[1], 0.0]

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
        self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
