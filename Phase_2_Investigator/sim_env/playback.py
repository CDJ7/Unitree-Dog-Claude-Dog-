"""Event playback (A5): animate how the event happened, in the simulated scene.

    pb = EventPlayback.from_ground_truth(scn, gt)          # exact: the scenario's own droplets
    pb = EventPlayback.from_descriptions(scn, events)      # Stage B: from inferred EventDescriptions
    pb.render_video("out.mp4", speed=0.25)                 # headless, slow motion
    pb.play_viewer(speed=0.25)                             # live in the MuJoCo viewer

What moves (all non-graphic: capsule mannequins, red droplets, red stains):
- an attacker mannequin's arm + weapon follow the swing: the recorded cast-off
  arc, or a downward strike ending at the impact origin;
- the victim stands at the event position and, for a drip trail, walks along it;
- each droplet flies its trajectory (gravity + drag, re-integrated from its
  launch, so it ends exactly on its stain) and its stain appears when it lands;
- smears grow along their path behind a hand marker;
- at the end the scene switches to the final positions (the crime scene).

Input is a list of `interfaces.EventDescription`. From ground truth they carry
the exact launches; from the analysis (Stage B) they carry mechanism, origin
and parameters, and the droplets are synthesised with the same launchers and
landed in the scene by ray casting.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from sim_env.interfaces import EventDescription
from sim_env.scenario_scene import (PREFIX_MANNEQUIN, PREFIX_STAIN, mannequin_parts, open_scenario)
from sim_env.spatter.constants import drag_k
from sim_env.spatter.physics import accel

PB = "pb_"
DROPLET_VIS_SCALE = 2.0          # droplets drawn 2x larger so they read in a video
GAP_S = 0.8                      # pause between events
STRIKE_S = 0.35                  # duration of a strike swing before the impact


@dataclass
class Track:
    """Piecewise-linear trajectory: times (K,), points (K, 3)."""
    t: np.ndarray
    p: np.ndarray

    def at(self, tq):
        tq = np.clip(tq, self.t[0], self.t[-1])
        return np.array([np.interp(tq, self.t, self.p[:, k]) for k in range(3)])


@dataclass
class Plan:
    duration: float
    droplets: list = field(default_factory=list)      # dicts: release, land, track, diameter, stain_geoms
    stains_at: dict = field(default_factory=dict)      # geom name -> appear time
    swings: list = field(default_factory=list)         # dicts: t0, t1, pivot, tip Track, kind
    walks: list = field(default_factory=list)          # (actor role, Track of feet xy)
    victim_falls_at: float = 1e9                       # victim switches to its final posture
    arm_rest_at: float = 1e9                           # attacker lowers the weapon
    hands: list = field(default_factory=list)          # smear hand marker: Track
    captions: list = field(default_factory=list)       # (t0, t1, text)


def _integrate(pos, vel, diameter, t_end, dt=1e-3):
    """Positions along a droplet flight (same physics as the generator)."""
    k = drag_k([diameter])
    x, v = np.array(pos, float)[None], np.array(vel, float)[None]
    ts, ps = [0.0], [x[0].copy()]
    t = 0.0
    while t < t_end:
        h = min(dt, t_end - t)
        vm = v + 0.5 * h * accel(v, k)
        x = x + h * vm
        v = v + h * accel(vm, k)
        t += h
        if len(ts) < 2 or t - ts[-1] >= 1 / 240 or t >= t_end:
            ts.append(t)
            ps.append(x[0].copy())
    return np.array(ts), np.array(ps)


def descriptions_from_ground_truth(gt):
    """One EventDescription per ground-truth event, with its exact launches."""
    out = []
    by_event = {}
    for d in gt["droplets"]:
        by_event.setdefault(d["event"], []).append(d)
    for ev in gt["events"]:
        launches = [{"t": d["release_time"], "pos": d["release_pos"], "vel": d["release_vel"],
                     "diameter": d["diameter"], "id": d["id"], "landed": d["landed"],
                     "flight_time": d["flight_time"], "impact_pos": d.get("impact_pos")}
                    for d in by_event.get(ev["index"], [])]
        out.append(EventDescription(
            mechanism=ev["mechanism"], origin=np.asarray(ev.get("origin") or [0, 0, 0], float),
            weapon_category=ev.get("weapon_category") or "none",
            swing_path=np.asarray(ev["swing_path"]) if ev.get("swing_path") else None,
            launches=launches, source="ground_truth", params=dict(ev.get("params") or {},
                                                                   swing_times=ev.get("swing_times"),
                                                                   end=ev.get("end"), smear_ids=ev.get("smear_ids"),
                                                                   actor=ev.get("actor") or "victim"),
            description=ev.get("description", "")))
    return out


# --------------------------------------------------------------------- scene
def _mocap_body(name, geoms_xml, pos=(0, 0, -5)):
    return f'<body name="{name}" mocap="true" pos="{pos[0]} {pos[1]} {pos[2]}">{geoms_xml}</body>'


def _figure_geoms(name, height, posture="standing", rgba="0.80 0.78 0.73 1", skip_right_arm=False):
    g = []
    for i, (typ, p, r) in enumerate(mannequin_parts(height, posture)):
        if skip_right_arm and i == 5:
            continue
        p = np.asarray(p)
        if typ == "sphere":
            g.append(f'<geom name="{name}_{i}" type="sphere" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" size="{r:.4f}" '
                     f'rgba="{rgba}" contype="0" conaffinity="0" group="2"/>')
        else:
            g.append(f'<geom name="{name}_{i}" type="capsule" fromto="{" ".join(f"{x:.4f}" for x in p)}" '
                     f'size="{r:.4f}" rgba="{rgba}" contype="0" conaffinity="0" group="2"/>')
    return "".join(g)


def playback_patch(scn, n_droplets, weapon_category):
    """Extra bodies for playback: event-pose victim, attacker + arm, droplet pool, hand marker."""
    mq = {m["role"]: m for m in scn.get("mannequins", [])}
    parts = []
    v = mq.get("victim", {"height": 1.75})
    parts.append(_mocap_body(f"{PB}victim", _figure_geoms(f"{PB}victim", v.get("height", 1.75),
                                                          v.get("event_posture", "standing"))))
    a = mq.get("attacker", {"height": 1.8})
    parts.append(_mocap_body(f"{PB}attacker", _figure_geoms(f"{PB}attacker", a.get("height", 1.8),
                                                            skip_right_arm=True, rgba="0.62 0.66 0.72 1")))
    edged = weapon_category == "edged"
    reach = 0.75 if edged else 0.95                       # swing radius of the weapon tip (mechanisms.py)
    if edged:   # blade from the hand to the tip
        weapon = (f'<geom name="pb_weapon" type="box" pos="{(0.45 + reach) / 2:.3f} 0 0" '
                  f'size="{(reach - 0.45) / 2:.3f} 0.014 0.003" rgba="0.75 0.77 0.80 1" '
                  f'contype="0" conaffinity="0" group="2"/>')
    else:       # pipe / bat
        weapon = (f'<geom name="pb_weapon" type="capsule" fromto="0.42 0 0 {reach:.3f} 0 0" size="0.022" '
                  f'rgba="0.35 0.25 0.18 1" contype="0" conaffinity="0" group="2"/>')
    arm = ('<geom name="pb_arm" type="capsule" fromto="0 0 0 0.45 0 0" size="0.05" rgba="0.62 0.66 0.72 1" '
           'contype="0" conaffinity="0" group="2"/>')
    parts.append(f'<body name="{PB}arm" mocap="true" pos="0 0 -5">{arm}{weapon}</body>')
    for i in range(n_droplets):
        parts.append(_mocap_body(f"{PB}d{i}", f'<geom name="{PB}dg{i}" type="sphere" size="0.003" '
                                              f'rgba="0.55 0.02 0.03 1" contype="0" conaffinity="0" group="2"/>'))
    parts.append(_mocap_body(f"{PB}hand", '<geom name="pb_hand" type="sphere" size="0.045" '
                                          'rgba="0.62 0.66 0.72 1" contype="0" conaffinity="0" group="2"/>'))
    xml = "".join(parts)

    def extra(root):
        from sim_env.scene_xml import append_xml
        append_xml(root, "worldbody", xml)
    return extra


def _quat_x_to(direction):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    x = np.array([1.0, 0, 0])
    c = np.cross(x, d)
    w = 1.0 + float(x @ d)
    if w < 1e-9:
        return np.array([0.0, 0, 0, 1.0])
    q = np.array([w, *c])
    return q / np.linalg.norm(q)


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


class EventPlayback:
    def __init__(self, scn, gt, events, label="GROUND TRUTH", fix_scale=True):
        self.scn, self.gt, self.events, self.label = scn, gt, events, label
        n = sum(len(e.launches) for e in events)
        cat = next((e.weapon_category for e in events if e.weapon_category in ("blunt", "edged")), "none")
        self.weapon_category = cat
        self.sim, _ = open_scenario(scn, extra=playback_patch(scn, n, cat))
        m = self.sim.model
        self.mocap = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b): int(m.body_mocapid[b])
                      for b in range(m.nbody) if m.body_mocapid[b] >= 0}
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self._names = names
        self.geom = {n: g for g, n in enumerate(names) if n}
        self.base_alpha = m.geom_rgba[:, 3].copy()
        self.final_figures = [g for g, n in enumerate(names) if n.startswith(PREFIX_MANNEQUIN)]
        self.stain_geoms = [g for g, n in enumerate(names) if n.startswith(PREFIX_STAIN)]
        for g, n in enumerate(names):
            if n.startswith(f"{PB}dg"):
                m.geom_size[g, 0] = 0.003
        self.plan = self._make_plan()

    # -------------------------------------------------------------- builders
    @classmethod
    def from_ground_truth(cls, scn, gt):
        return cls(scn, gt, descriptions_from_ground_truth(gt), label="GROUND TRUTH")

    @classmethod
    def from_descriptions(cls, scn, gt, events, label="INFERRED EVENT"):
        """Synthesise droplets for descriptions without launches (Stage B), then play them."""
        from sim_env.scenario_scene import SimSession, _props_patch
        from sim_env.spatter import generate
        from sim_env.spatter.mj_surfaces import MjSurfaces
        need = [e for e in events if not e.launches and e.mechanism != "transfer"]
        if need:
            evs = [{"mechanism": e.mechanism, "origin": np.asarray(e.origin).tolist(),
                    "weapon_category": e.weapon_category,
                    "params": {k: v for k, v in e.params.items() if k not in ("swing_times", "smear_ids")}}
                   for e in need]
            s2 = dict(copy.deepcopy(scn), events=evs, summary={"primary_event": 0})
            sim = SimSession(scn["environment"], scene_patch=_props_patch(scn), spawn=scn.get("spawn"))
            try:
                exclude = set(sim.robot_geoms) | {g for g in range(sim.model.ngeom) if (mujoco.mj_id2name(
                    sim.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(PREFIX_MANNEQUIN)}
                syn = generate(s2, surfaces=MjSurfaces(sim.model, sim.data, exclude), satellites=False)
            finally:
                sim.close()
            for e, d_ev in zip(need, descriptions_from_ground_truth(syn)):
                e.launches = [dict(l, id=f"syn_{l['id']}") for l in d_ev.launches]
                if e.swing_path is None:
                    e.swing_path = d_ev.swing_path
                    e.params.setdefault("swing_times", d_ev.params.get("swing_times"))
        return cls(scn, gt, events, label=label)

    def _make_plan(self):
        gt, scn = self.gt, self.scn
        mq = {m["role"]: m for m in scn.get("mannequins", [])}
        att = mq.get("attacker")
        plan = Plan(duration=0.0)
        t = 1.0
        di = 0
        for ei, ev in enumerate(self.events):
            t_ev = t
            if ev.mechanism == "impact" and ev.weapon_category in ("blunt", "edged") and att:
                t_ev = t + STRIKE_S
                h = att.get("height", 1.8)
                pivot = np.r_[att["event_position"][:2], 1.45 * h / 1.8]
                o = np.asarray(ev.origin, float)
                r = float(np.linalg.norm(o - pivot))
                fwd = (o - pivot)
                heading = np.arctan2(fwd[1], fwd[0])
                f = np.array([np.cos(heading), np.sin(heading), 0.0])
                phi_end = np.arcsin(np.clip(fwd[2] / max(r, 1e-6), -1, 1))
                phis = np.linspace(np.radians(115), phi_end, 12)
                tip = np.array([pivot + r * (np.cos(p) * f + np.sin(p) * np.array([0, 0, 1])) for p in phis])
                plan.swings.append({"t0": t, "t1": t_ev, "pivot": pivot, "track": Track(np.linspace(t, t_ev, 12), tip)})
            if ev.mechanism == "castoff" and ev.swing_path is not None:
                st = ev.params.get("swing_times") or np.linspace(0, 0.5, len(ev.swing_path))
                plan.swings.append({"t0": t_ev, "t1": t_ev + st[-1], "pivot": np.asarray(ev.origin, float),
                                    "track": Track(t_ev + np.asarray(st), np.asarray(ev.swing_path))})
            if ev.mechanism == "drip" and ev.params.get("end") is not None:
                start, end = np.asarray(ev.origin, float), np.asarray(ev.params["end"], float)
                rel = [l["t"] for l in ev.launches] or [0.0]
                plan.walks.append((ev.params.get("actor", "victim"),
                                   Track(np.array([t_ev, t_ev + max(rel) + 0.2]), np.array([start, end]))))
            end_t = t_ev
            for l in ev.launches:
                ft = float(l.get("flight_time", 1.0))
                ts, ps = _integrate(l["pos"], l["vel"], l["diameter"], ft)
                if l.get("landed") and l.get("impact_pos") is not None:
                    ps[-1] = l["impact_pos"]
                rel = t_ev + float(l["t"])
                trk = Track(rel + ts, ps)
                prefix = f"{PREFIX_STAIN}{l['id']}_st"
                geoms = [g for n, g in self.geom.items() if n == prefix or n.startswith(prefix + "_")]
                plan.droplets.append({"i": di, "release": rel, "land": rel + ft, "track": trk,
                                      "diameter": l["diameter"], "landed": bool(l.get("landed", True))})
                for g in geoms:
                    plan.stains_at[g] = min(plan.stains_at.get(g, 1e9), rel + ft)
                end_t = max(end_t, rel + ft)
                di += 1
            if ev.mechanism == "transfer":
                for sid in ev.params.get("smear_ids") or []:
                    sm = next(s for s in gt["smears"] if s["id"] == sid)
                    pts = np.asarray(sm["path_world"], float)
                    dur = 1.2
                    plan.hands.append(Track(np.linspace(t_ev, t_ev + dur, len(pts)), pts + 0.03 * np.asarray(
                        next(s for s in gt["surfaces"] if s["id"] == sm["surface_id"])["normal"])))
                    pre = f"{PREFIX_STAIN}{sid}_"
                    for n, g in self.geom.items():
                        if n.startswith(pre):
                            k = int(n[len(pre):].split("_")[0])
                            plan.stains_at[g] = t_ev + dur * k / max(len(pts) - 1, 1)
                    end_t = max(end_t, t_ev + dur)
            for s in plan.swings:
                end_t = max(end_t, s["t1"])
            plan.captions.append((t, end_t + GAP_S, f"{ei + 1}. {ev.description or ev.mechanism}"))
            if ev.mechanism in ("impact", "projected") and plan.victim_falls_at > 1e8:
                plan.victim_falls_at = end_t + 0.3
            if ev.mechanism == "drip" and ev.params.get("actor", "victim") == "victim":
                plan.victim_falls_at = end_t + 0.3                # walked first, then collapses
            t = end_t + GAP_S
        plan.duration = t + 1.5
        plan.final_at = t
        plan.victim_falls_at = min(plan.victim_falls_at, t)
        if plan.swings:
            plan.arm_rest_at = max(s["t1"] for s in plan.swings) + 0.4
        return plan

    # ------------------------------------------------------------------ pose
    def set_time(self, tq):
        """Put every animated body and stain in its state at playback time tq."""
        m, d, P = self.sim.model, self.sim.data, self.plan
        mq = {x["role"]: x for x in self.scn.get("mannequins", [])}
        final = tq >= P.final_at
        fallen = tq >= P.victim_falls_at
        # final crime scene vs event figures (the victim collapses after the blow / its walk)
        for g in self.final_figures:
            is_victim = self._names[g].startswith(f"{PREFIX_MANNEQUIN}victim")
            m.geom_rgba[g, 3] = self.base_alpha[g] if (final or (is_victim and fallen)) else 0.0
        v = mq.get("victim")
        vid = self.mocap[f"{PB}victim"]
        if v and not fallen:
            xy = np.asarray(v["event_position"][:2], float)
            for role, w in P.walks:
                if role == "victim" and w.t[0] <= tq:
                    xy = w.at(tq)[:2]
            d.mocap_pos[vid] = [xy[0], xy[1], 0.0]
            d.mocap_quat[vid] = _yaw_quat(np.radians(v.get("event_yaw_deg", 0)))
        else:
            d.mocap_pos[vid] = [0, 0, -5]
        a = mq.get("attacker")
        aid, arm = self.mocap[f"{PB}attacker"], self.mocap[f"{PB}arm"]
        if a and not final and P.swings:
            axy = np.asarray(a["event_position"][:2], float)
            moved = np.zeros(2)
            for role, w in P.walks:
                if role == "attacker" and w.t[0] <= tq:
                    moved = w.at(tq)[:2] - w.p[0, :2]
            axy = axy + moved
            yaw = np.radians(a.get("event_yaw_deg", 0))
            d.mocap_pos[aid] = [axy[0], axy[1], 0.0]
            d.mocap_quat[aid] = _yaw_quat(yaw)
            if tq >= P.arm_rest_at:             # weapon lowered, hanging forward-down
                d.mocap_pos[arm] = np.r_[P.swings[-1]["pivot"][:2] + moved, P.swings[-1]["pivot"][2]]
                d.mocap_quat[arm] = _quat_x_to([0.35 * np.cos(yaw), 0.35 * np.sin(yaw), -1.0])
            else:
                sw = [s for s in P.swings if s["t0"] - 0.6 <= tq] or P.swings[:1]
                s = sw[-1]
                tip = s["track"].at(tq)
                d.mocap_pos[arm] = s["pivot"]
                d.mocap_quat[arm] = _quat_x_to(tip - s["pivot"])
        else:
            d.mocap_pos[aid] = [0, 0, -5]
            d.mocap_pos[arm] = [0, 0, -5]
        # droplets in flight
        for dr in P.droplets:
            bid = self.mocap[f"{PB}d{dr['i']}"]
            if dr["release"] <= tq < dr["land"] and not final:
                d.mocap_pos[bid] = dr["track"].at(tq)
            else:
                d.mocap_pos[bid] = [0, 0, -5]
        # stains appear when they land
        for g in self.stain_geoms:
            m.geom_rgba[g, 3] = self.base_alpha[g] if (final or P.stains_at.get(g, 0.0) <= tq) else 0.0
        hid = self.mocap[f"{PB}hand"]
        d.mocap_pos[hid] = [0, 0, -5]
        for h in P.hands:
            if h.t[0] <= tq <= h.t[-1] + 0.2 and not final:
                d.mocap_pos[hid] = h.at(tq)
        mujoco.mj_forward(m, d)

    def caption(self, tq):
        for t0, t1, text in self.plan.captions:
            if t0 <= tq < t1:
                return text
        return "Final scene" if tq >= self.plan.final_at else ""

    # ---------------------------------------------------------------- output
    def view_camera(self, distance=3.2, elevation=-20.0):
        """Free camera inside the room, looking at the event from the robot's side."""
        pts = [e.origin for e in self.events if e.origin is not None]
        c = np.mean(pts, axis=0) if pts else np.zeros(3)
        st = np.array([s["center_world"] for s in self.gt["stains"]] or [c])
        target = 0.6 * c + 0.4 * st.mean(axis=0)
        spawn = np.asarray(self.scn.get("spawn", [0, 0])[:2], float)
        d = spawn - target[:2]
        mq = {x["role"]: x for x in self.scn.get("mannequins", [])}
        if "attacker" in mq and "victim" in mq and self.plan.swings:
            # side view of the attacker -> victim line, so neither hides the other
            line = np.asarray(mq["victim"]["event_position"][:2], float) - \
                np.asarray(mq["attacker"]["event_position"][:2], float)
            perp = np.array([-line[1], line[0]])
            d = perp if perp @ d >= 0 else -perp
            mid = 0.5 * (np.asarray(mq["victim"]["event_position"][:2]) + np.asarray(mq["attacker"]["event_position"][:2]))
            target = np.array([mid[0], mid[1], target[2]])
        d = d / max(np.linalg.norm(d), 1e-6)
        # keep the camera inside the room: stop short of the first wall/furniture behind it
        best = None
        side_view = "attacker" in mq and "victim" in mq and self.plan.swings
        for off in ((0, 180, 20, -20, 160, -160, 40, -40) if side_view else (0, 20, -20, 40, -40, 60, -60)):
            a = np.radians(off)
            dd = np.array([np.cos(a) * d[0] - np.sin(a) * d[1], np.sin(a) * d[0] + np.cos(a) * d[1]])
            vec = np.array([dd[0], dd[1], -np.tan(np.radians(elevation))])
            vec /= np.linalg.norm(vec)
            gid = np.zeros(1, np.int32)
            hit = mujoco.mj_ray(self.sim.model, self.sim.data, np.array([target[0], target[1], 0.9]), vec,
                                np.array([1, 1, 0, 0, 0, 0], np.uint8), 1, -1, gid)
            free = distance if hit < 0 else min(distance, hit - 0.25)
            if free >= 1.8:                     # good enough: keep the preferred direction
                best = (free, dd)
                break
            if best is None or free > best[0] + 0.3:
                best = (free, dd)
        dist, dd = best
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [target[0], target[1], 0.9]
        cam.distance = max(dist, 1.2)
        cam.azimuth = float(np.degrees(np.arctan2(-dd[1], -dd[0])))
        cam.elevation = elevation
        return cam

    def render_video(self, path, speed=0.25, fps=30, size=(960, 544), camera=None):
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        cam = camera or self.view_camera()
        az0 = cam.azimuth
        n = int(self.plan.duration / speed * fps)
        frames = []
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=8) \
            if str(path).endswith(".mp4") else None
        for k in range(n):
            tq = k / fps * speed
            self.set_time(tq)
            cam.azimuth = az0 + 25 * np.sin(2 * np.pi * tq / self.plan.duration)
            img = Image.fromarray(self.sim.render_view(size[0], size[1], camera=cam))
            dr = ImageDraw.Draw(img)
            dr.rectangle([0, 0, size[0], 26], fill=(0, 0, 0))
            dr.text((8, 7), f"{self.label}  |  {self.scn['id']}  |  t = {tq:5.2f} s  ({speed:g}x)  |  "
                            f"{self.caption(tq)}", fill=(255, 255, 255))
            arr = np.asarray(img)
            if writer:
                writer.append_data(arr)
            else:
                frames.append(img)
        if writer:
            writer.close()
        elif frames:
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
        return path

    def snapshot(self, tq, size=(960, 544), camera=None):
        self.set_time(tq)
        return self.sim.render_view(size[0], size[1], camera=camera or self.view_camera())

    def play_viewer(self, speed=0.25, loop=True):
        import mujoco.viewer
        v = mujoco.viewer.launch_passive(self.sim.model, self.sim.data, show_left_ui=False, show_right_ui=False)
        cam = self.view_camera()
        with v.lock():
            v.opt.geomgroup[3] = 0
            v.opt.geomgroup[4] = 0
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.lookat[:], v.cam.distance = cam.lookat, cam.distance
            v.cam.azimuth, v.cam.elevation = cam.azimuth, cam.elevation
        t0 = time.monotonic()
        last = ""
        try:
            while v.is_running():
                tq = (time.monotonic() - t0) * speed
                if tq > self.plan.duration:
                    if not loop:
                        break
                    t0 = time.monotonic()
                    continue
                with v.lock():
                    self.set_time(tq)
                cap = self.caption(tq)
                if cap != last:
                    print(f"[t={tq:5.2f}s] {cap}", flush=True)
                    last = cap
                v.sync()
                time.sleep(1 / 60)
        finally:
            v.close()

    def close(self):
        self.sim.close()
