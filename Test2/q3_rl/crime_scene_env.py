"""Q3: CrimeSceneSearchEnv, a Gymnasium environment for evidence search with a Go2 in house 185.

The world is the real MuJoCo crime scene reduced to 2-D (q3_rl/extract_layouts.py): the collision
footprint raster of walls/furniture/props (5 cm cells) and the ground-truth stain clusters (the
evidence). The dog is a disc with simple kinematics: the action is the same body-frame velocity
command (vx, vy, yaw_rate) the Q2 controller takes, tracked with a first-order lag. A policy trained
here is deployed unchanged on the MuJoCo Go2 through CrimeSceneGo2.set_velocity_command
(q3_rl/deploy_mujoco.py).

Evidence is FOUND when it is inside the head camera's view: within `find.range` of the camera, inside
its horizontal field of view and with a clear line of sight on the raster. Walking over a FLOOR stain
(footprint disc overlaps the stain cluster) CONTAMINATES the scene. Bumping into an obstacle is a
COLLISION: the dog's body is the Go2's 0.70 x 0.32 m rectangle (robot.footprint; a 0.24 m disc if absent),
and a translation or turn that would make it overlap an obstacle cell is cancelled.

Observation (54 floats in [-1, 1]):
  0-3   pose: x, y (normalised to the house), sin(yaw), cos(yaw)
  4-6   body velocity vx, vy, yaw_rate (normalised by the limits)
  7-30  lidar: 24 rays over 360 deg (ray 0 straight ahead, counter-clockwise), range / max_range
  31-36 nearest unfound evidence (nearest by path length): body-frame dx, dy (/5 m), straight-line
        distance (/5 m), path length on the map (/10 m), height (/2 m), is-floor flag
  37-39 nearest floor stain: body-frame dx, dy (/3 m), clearance between footprint and stain (/3 m)
  40-47 found flags for up to 8 evidence slots (1 = found or empty slot, 0 = still to find)
  48-49 route hint: body-frame unit vector along the shortest path on the known floor plan towards
        the nearest unfound evidence (what the map + planner of a real robot would give; it says
        nothing about stains, so contamination avoidance and the visiting order are left to RL)
  50-52 previous action
  53    bumped flag: 1 if the last step was a collision (so the policy can tell it is stuck)
Action: Box(-1, 1, (3,)) scaled to the Go2 limits (0.4 m/s, 0.15 m/s, 0.5 rad/s).
Reward: sum of the terms below (each its own function; weights in config/q3_env.yaml):
  find, progress, contamination, collision, time, all_found.
Episode ends (terminated) when all evidence is found (success) or on a severe collision: the dog
has been pushing into an obstacle for episode.severe_collision_steps consecutive steps (failure);
truncated after episode.max_steps.
Domain randomisation on every reset: layout, spawn pose, evidence subset and jitter, velocity lag,
speed gain, action noise, lidar noise and dropout.
"""
from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from scipy import ndimage

TEST2 = Path(__file__).resolve().parents[1]
LAYOUT_DIR = Path(__file__).resolve().parent / "layouts"
HOUSE = (0.0, 8.2, 0.0, 4.5)          # x0, x1, y0, y1 of house 185
N_OBS = 54
HINT_ANGLES = np.arange(16) * 2 * np.pi / 16


def load_env_config(path=None):
    return yaml.safe_load(Path(path or TEST2 / "config/q3_env.yaml").read_text())


# ---------------------------------------------------------------------------- reward terms
def reward_find(w, n_new):
    """+w for every evidence item seen for the first time this step."""
    return w["find"] * n_new


def reward_progress(w, d_before, d_after):
    """+w per metre the path length to the nearest unfound item shrank (negative if it grew)."""
    if not (np.isfinite(d_before) and np.isfinite(d_after)):
        return 0.0
    return w["progress"] * float(np.clip(d_before - d_after, -0.5, 0.5))


def reward_contamination(w, on_stain, entered):
    """-w per step standing on a floor stain, and a one-off penalty when stepping onto a new one."""
    return -w["contamination"] * float(on_stain) - w["contamination_entry"] * entered


def reward_collision(w, collided, severe=False):
    """-w per step in which the dog bumped into a wall or piece of furniture, and a one-off
    penalty when the episode ends because of a severe collision."""
    return -w["collision"] * float(collided) - w["severe_collision"] * float(severe)


def reward_time(w):
    """-w every step: search quickly."""
    return -w["time"]


def reward_all_found(w, done):
    """+w once, when the last item is found."""
    return w["all_found"] * float(done)


# ---------------------------------------------------------------------------- layouts
class Layout:
    """One exported scene with everything the env needs precomputed (lazily, once per process)."""

    def __init__(self, blocked, meta, origin, res, cfg):
        self.name = meta["name"]
        self.blocked = blocked
        self.origin = np.asarray(origin, float)
        self.res = float(res)
        self.H, self.W = blocked.shape
        rc = cfg["robot"]
        self.clear = ndimage.distance_transform_edt(~blocked) * self.res
        ys = self.origin[1] + (np.arange(self.H) + 0.5) * self.res
        xs = self.origin[0] + (np.arange(self.W) + 0.5) * self.res
        inside = ((ys[:, None] > HOUSE[2]) & (ys[:, None] < HOUSE[3]) & (xs[None] > HOUSE[0]) & (xs[None] < HOUSE[1]))
        # cells the body's centre may occupy: a disc of `radius`, or with a rectangular `footprint` any cell
        # with clearance >= half its width (it may then only pass lengthwise; the exact test is in step())
        r_reach = min(rc["radius"], rc["footprint"][1] / 2) if rc.get("footprint") else rc["radius"]
        free = (self.clear >= r_reach) & inside
        lab, n = ndimage.label(free, structure=np.ones((3, 3)))
        sizes = ndimage.sum(free, lab, range(1, n + 1))
        self.reach = lab == (1 + int(np.argmax(sizes)))
        self.spawnable = self.reach & (self.clear >= cfg["episode"]["spawn_clearance"])
        ev = meta["evidence"]
        ec = cfg["evidence"]
        self.floor = np.array([[*e["center"][:2], e["radius"]] for e in ev if e["surface_kind"] == "floor"]).reshape(-1, 3)
        self.items = [e for e in ev if e["n_stains"] >= ec["min_stains"] and e["center"][2] <= ec["max_height"]]
        self._fields = {}
        keep = [i for i in range(len(self.items)) if np.isfinite(self.field(i)).any()]
        self.dropped = [self.items[i]["id"] for i in range(len(self.items)) if i not in keep]
        self._fields = {(j, False): self._fields[(i, False)] for j, i in enumerate(keep)}
        self.items = [self.items[i] for i in keep]

    def cell(self, xy):
        ij = np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(int)
        return int(np.clip(ij[1], 0, self.H - 1)), int(np.clip(ij[0], 0, self.W - 1))

    def cell_xy(self, r, c):
        return self.origin + (np.array([c, r]) + 0.5) * self.res

    def line_of_sight(self, a, b, stop=0.35):
        """True if the raster is clear along a->b, ignoring the last `stop` m (the target's own surface)."""
        d = np.linalg.norm(b - a)
        n = int(max(d - stop, 0) / (0.5 * self.res)) + 1
        t = np.linspace(0, max(d - stop, 0) / max(d, 1e-9), n)
        p = a[None] + t[:, None] * (b - a)[None]
        ij = np.floor((p - self.origin) / self.res).astype(int)
        ok = (ij[:, 1] >= 0) & (ij[:, 1] < self.H) & (ij[:, 0] >= 0) & (ij[:, 0] < self.W)
        if not ok.all():
            return False
        return not self.blocked[ij[:, 1], ij[:, 0]].any()

    def field(self, k, find_range=1.5, soft=False):
        """Path length (m) from every reachable cell to the nearest cell the item can be seen from.
        soft=True: clearance-weighted path cost (1 + 4 * ((0.6 - clearance) / 0.36)^2 near obstacles, as
        in the Q2 planner), whose descent direction runs down the middle of passages (route hint)."""
        key = (k, soft)
        if key in self._fields:
            return self._fields[key]
        from skimage.graph import MCP_Geometric
        tgt = np.asarray(self.items[k]["center"][:2], float)
        rr, cc = np.nonzero(self.reach)
        xy = self.origin + (np.stack([cc, rr], 1) + 0.5) * self.res
        near = np.linalg.norm(xy - tgt, axis=1) <= find_range
        seeds = [(r, c) for (r, c), p in zip(zip(rr[near], cc[near]), xy[near]) if self.line_of_sight(p, tgt)]
        f = np.full((self.H, self.W), np.inf)
        if seeds:
            cost = np.where(self.reach, 1.0, np.inf)
            if soft:
                cost = cost + 4.0 * np.clip((0.6 - self.clear) / 0.36, 0, 1) ** 2
            mcp = MCP_Geometric(cost, fully_connected=True)
            cum, _ = mcp.find_costs(seeds)
            f = np.where(self.reach, cum * self.res, np.inf)
        self._fields[key] = f
        return f


_LAYOUT_CACHE = {}


def load_layouts(names, cfg):
    meta = json.loads((LAYOUT_DIR / "house185_layouts.json").read_text())
    blocked = np.load(LAYOUT_DIR / "house185_layouts.npz")["blocked"]
    out = []
    for n in names:
        key = (n, json.dumps(cfg["evidence"]), json.dumps(cfg["robot"]))
        if key not in _LAYOUT_CACHE:
            i = meta["names"].index(n)
            _LAYOUT_CACHE[key] = Layout(blocked[i], meta["layouts"][i], meta["origin"], meta["res"], cfg)
        out.append(_LAYOUT_CACHE[key])
    return out


# ---------------------------------------------------------------------------- environment
class CrimeSceneSearchEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 5}

    def __init__(self, split="train", config=None, layouts=None, randomize=True, render_mode=None):
        self.cfg = config if isinstance(config, dict) else load_env_config(config)
        names = layouts or self.cfg["layouts"][split]
        self.layouts = [l for l in load_layouts(names, self.cfg) if l.items]
        self.randomize = randomize
        self.render_mode = render_mode
        rc = self.cfg["robot"]
        self.limits = np.asarray(rc["limits"], float)
        self.w = self.cfg["reward"]
        self.max_items = self.cfg["evidence"]["max_items"]
        self.n_rays = self.cfg["lidar"]["rays"]
        self.ray_angles = np.arange(self.n_rays) * 2 * np.pi / self.n_rays
        self.ray_steps = np.arange(0.1, self.cfg["lidar"]["max_range"] + 1e-6, 0.05)
        fp = rc.get("footprint")
        if fp:                                  # body-frame sample points covering the Go2's rectangle
            xs = np.linspace(-fp[0] / 2, fp[0] / 2, int(np.ceil(fp[0] / 0.04)) + 1)
            ys = np.linspace(-fp[1] / 2, fp[1] / 2, int(np.ceil(fp[1] / 0.04)) + 1)
            self.body_pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
        else:
            self.body_pts = None
        self.observation_space = spaces.Box(-1.0, 1.0, (N_OBS,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.layout = None

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        L = self.layouts[options["layout"]] if "layout" in options else self.layouts[rng.integers(len(self.layouts))]
        self.layout = L
        ec, rc = self.cfg["evidence"], self.cfg["robot"]
        n = len(L.items)
        if self.randomize and n > ec["subset_min"]:
            k = int(rng.integers(ec["subset_min"], n + 1))
            idx = np.sort(rng.choice(n, size=min(k, self.max_items), replace=False))
        else:
            idx = np.arange(min(n, self.max_items))
        self.items = idx
        self.tgt = np.array([L.items[i]["center"] for i in idx], float)
        if self.randomize:
            self.tgt[:, :2] += rng.uniform(-ec["jitter"], ec["jitter"], (len(idx), 2))
        self.is_floor = np.array([L.items[i]["surface_kind"] == "floor" for i in idx])
        self.found = np.zeros(len(idx), bool)
        # spawn: a free cell with clearance, not on a stain and not already seeing an item
        rr, cc = np.nonzero(L.spawnable)
        for _ in range(200):
            j = rng.integers(len(rr))
            xy = L.cell_xy(rr[j], cc[j]) + rng.uniform(-0.02, 0.02, 2)
            yaw = float(rng.uniform(-np.pi, np.pi))
            if "spawn" in options:
                xy, yaw = np.asarray(options["spawn"][:2], float), float(options["spawn"][2])
            self.xy, self.yaw = xy, yaw
            if self._on_stain() < 0 and not self._visible().any() and self._fits(self.xy, self.yaw):
                break
        self.v = np.zeros(3)
        self.last_action = np.zeros(3)
        self.last_collided = False
        r = self.randomize
        self.tau = float(rng.uniform(*rc["lag_tau"])) if r else float(np.mean(rc["lag_tau"]))
        self.gain = float(rng.uniform(*rc["speed_scale"])) if r else 1.0
        self.act_noise = rc["action_noise"] if r else 0.0
        self.lidar_noise = self.cfg["lidar"]["noise"] if r else 0.0
        self.lidar_drop = self.cfg["lidar"]["dropout"] if r else 0.0
        self.t = 0
        self.stats = {"collisions": 0, "contamination_steps": 0, "contamination_entries": 0, "path_m": 0.0,
                      "severe_collision": False}
        self.coll_run = 0
        self.was_on = self._on_stain()
        self.d_near = self._nearest_unfound()[1]
        self.traj = [(*self.xy, self.yaw)]
        return self._obs(), self._info()

    # ------------------------------------------------------------------ geometry helpers
    def _fits(self, xy, yaw):
        """True if the robot at (xy, yaw) overlaps no obstacle cell: the rotated footprint rectangle
        (robot.footprint) or, without one, the disc of robot.radius (its centre on a reachable cell)."""
        L = self.layout
        if self.body_pts is None:
            r, c = L.cell(xy)
            return bool(L.reach[r, c])
        c, s = np.cos(yaw), np.sin(yaw)
        p = xy[None] + self.body_pts @ np.array([[c, s], [-s, c]])
        ij = np.floor((p - L.origin) / L.res).astype(int)
        if (ij[:, 1] < 0).any() or (ij[:, 1] >= L.H).any() or (ij[:, 0] < 0).any() or (ij[:, 0] >= L.W).any():
            return False
        r0, c0 = L.cell(xy)
        return bool(L.reach[r0, c0]) and not L.blocked[ij[:, 1], ij[:, 0]].any()

    def _on_stain(self):
        """Index of a floor stain under the footprint, or -1."""
        F = self.layout.floor
        if not len(F):
            return -1
        d = np.linalg.norm(F[:, :2] - self.xy, axis=1) - (self.cfg["robot"]["foot_radius"] + F[:, 2])
        k = int(np.argmin(d))
        return k if d[k] < 0 else -1

    def _visible(self):
        fc = self.cfg["find"]
        cam = self.xy + fc["camera_offset"] * np.array([np.cos(self.yaw), np.sin(self.yaw)])
        vis = np.zeros(len(self.tgt), bool)
        for i, t in enumerate(self.tgt):
            if self.found[i]:
                continue
            d = t[:2] - cam
            dist = np.linalg.norm(d)
            if dist > fc["range"]:
                continue
            bearing = np.arctan2(d[1], d[0]) - self.yaw
            bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
            if dist > 0.25 and abs(bearing) > np.radians(fc["half_fov_deg"]):
                continue
            vis[i] = self.layout.line_of_sight(cam, t[:2]) if dist > 0.4 else True
        return vis

    def _path_len(self, i):
        r, c = self.layout.cell(self.xy)
        return float(self.layout.field(int(self.items[i]))[r, c])

    def _nearest_unfound(self):
        best, bd = -1, np.inf
        for i in np.nonzero(~self.found)[0]:
            d = self._path_len(i)
            if d < bd or best < 0:
                best, bd = int(i), d
        return best, bd

    def _lidar(self):
        L = self.layout
        ang = self.yaw + self.ray_angles
        p = self.xy[None, None] + self.ray_steps[None, :, None] * np.stack([np.cos(ang), np.sin(ang)], -1)[:, None]
        ij = np.floor((p - L.origin) / L.res).astype(int)
        r, c = ij[..., 1], ij[..., 0]
        out = (r < 0) | (r >= L.H) | (c < 0) | (c >= L.W)
        hit = out | L.blocked[np.clip(r, 0, L.H - 1), np.clip(c, 0, L.W - 1)]
        first = np.where(hit.any(1), hit.argmax(1), len(self.ray_steps) - 1)
        rng = self.ray_steps[first]
        rng = np.where(hit.any(1), rng, self.cfg["lidar"]["max_range"])
        if self.lidar_noise:
            rng = rng + self.np_random.normal(0, self.lidar_noise, rng.shape)
            rng = np.where(self.np_random.random(rng.shape) < self.lidar_drop, self.cfg["lidar"]["max_range"], rng)
        return np.clip(rng, 0, self.cfg["lidar"]["max_range"])

    def _route_hint(self, i, step=0.35):
        """Unit vector (body frame) of the steepest descent of item i's path-length field."""
        L = self.layout
        f = L.field(int(self.items[i]), soft=True)
        r, c = L.cell(self.xy)
        if not np.isfinite(f[r, c]) or f[r, c] <= 0.0:
            d = self.tgt[i, :2] - self.xy
            return self._body(d / max(np.linalg.norm(d), 1e-6))
        p = self.xy[None] + step * np.stack([np.cos(HINT_ANGLES), np.sin(HINT_ANGLES)], 1)
        ij = np.floor((p - L.origin) / L.res).astype(int)
        vals = f[np.clip(ij[:, 1], 0, L.H - 1), np.clip(ij[:, 0], 0, L.W - 1)]
        k = int(np.argmin(vals))
        if not np.isfinite(vals[k]):
            return np.zeros(2)
        return self._body(np.array([np.cos(HINT_ANGLES[k]), np.sin(HINT_ANGLES[k])]))

    def _body(self, d):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    # ------------------------------------------------------------------ observation
    def _obs(self):
        o = np.zeros(N_OBS, np.float32)
        o[0] = (self.xy[0] - 4.1) / 4.1
        o[1] = (self.xy[1] - 2.25) / 2.25
        o[2], o[3] = np.sin(self.yaw), np.cos(self.yaw)
        o[4:7] = self.v / self.limits
        o[7:31] = self._lidar() / self.cfg["lidar"]["max_range"]
        i, path = self._nearest_unfound()
        if i >= 0:
            d = self.tgt[i, :2] - self.xy
            b = self._body(d)
            o[31:33] = b / 5.0
            o[33] = np.linalg.norm(d) / 5.0
            o[34] = (path if np.isfinite(path) else 10.0) / 10.0
            o[35] = self.tgt[i, 2] / 2.0
            o[36] = float(self.is_floor[i])
        F = self.layout.floor
        if len(F):
            d = F[:, :2] - self.xy
            gap = np.linalg.norm(d, axis=1) - (self.cfg["robot"]["foot_radius"] + F[:, 2])
            k = int(np.argmin(gap))
            o[37:39] = self._body(d[k]) / 3.0
            o[39] = gap[k] / 3.0
        else:
            o[37:40] = 1.0
        flags = np.ones(self.max_items, np.float32)
        flags[:len(self.found)] = self.found
        o[40:48] = flags
        if i >= 0:
            o[48:50] = self._route_hint(i)
        o[50:53] = self.last_action
        o[53] = float(self.last_collided)
        return np.clip(o, -1, 1).astype(np.float32)

    def _info(self):
        return {"layout": self.layout.name, "n_items": int(len(self.found)), "n_found": int(self.found.sum()),
                "success": bool(self.found.all()), "is_success": bool(self.found.all()), "time_s": self.t * self.cfg["robot"]["dt"], **self.stats}

    # ------------------------------------------------------------------ step
    def step(self, action):
        rc = self.cfg["robot"]
        a = np.clip(np.asarray(action, float), -1, 1)
        cmd = a * self.limits * self.gain
        if self.act_noise:
            cmd = cmd + self.np_random.normal(0, self.act_noise, 3) * self.limits
        h = rc["dt"] / rc["substeps"]
        alpha = 1 - np.exp(-h / self.tau)
        unfound_before = ~self.found
        collided = False
        for _ in range(rc["substeps"]):
            self.v += (cmd - self.v) * alpha
            c, s = np.cos(self.yaw), np.sin(self.yaw)
            nxt = self.xy + h * np.array([c * self.v[0] - s * self.v[1], s * self.v[0] + c * self.v[1]])
            nyaw = (self.yaw + h * self.v[2] + np.pi) % (2 * np.pi) - np.pi
            if self._fits(nxt, nyaw):
                pass
            elif self._fits(self.xy, nyaw):          # the translation is blocked, the turn is not
                nxt, collided = self.xy, True
                self.v[:2] = 0.0
            elif self._fits(nxt, self.yaw):          # the turn is blocked (body would swing into something)
                nyaw, collided = self.yaw, True
                self.v[2] = 0.0
            else:
                nxt, nyaw, collided = self.xy, self.yaw, True
                self.v[:] = 0.0
            self.stats["path_m"] += float(np.linalg.norm(nxt - self.xy))
            self.xy, self.yaw = nxt, nyaw
        self.t += 1
        # progress is measured against the items that were unfound before this step
        d_after = min((self._path_len(i) for i in np.nonzero(unfound_before)[0]), default=np.inf)
        d_before = self.d_near
        new = self._visible()
        self.found |= new
        n_new = int(new.sum())
        on = self._on_stain()
        entered = int(on >= 0 and on != self.was_on)
        self.was_on = on
        done = bool(self.found.all())
        self.coll_run = self.coll_run + 1 if collided else 0
        severe = (not done) and self.coll_run >= self.cfg["episode"]["severe_collision_steps"]
        self.stats["severe_collision"] = bool(severe)
        self.stats["collisions"] += int(collided)
        self.stats["contamination_steps"] += int(on >= 0)
        self.stats["contamination_entries"] += entered
        terms = {"find": reward_find(self.w, n_new),
                 "progress": reward_progress(self.w, d_before, d_after),
                 "contamination": reward_contamination(self.w, on >= 0, entered),
                 "collision": reward_collision(self.w, collided, severe),
                 "time": reward_time(self.w),
                 "all_found": reward_all_found(self.w, done)}
        self.d_near = self._nearest_unfound()[1] if not done else 0.0
        self.traj.append((*self.xy, self.yaw))
        self.last_action = a.copy()
        self.last_collided = collided
        truncated = (not done) and (not severe) and self.t >= self.cfg["episode"]["max_steps"]
        info = self._info()
        info["reward_terms"] = terms
        info["collided"], info["on_stain"], info["new_found"] = collided, bool(on >= 0), n_new
        return self._obs(), float(sum(terms.values())), done or severe, truncated, info

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        L, s = self.layout, 3
        img = np.where(L.blocked[..., None], np.array([70, 70, 70], np.uint8), np.array([235, 232, 225], np.uint8))
        img[L.reach & ~L.blocked] = (250, 250, 245)

        def disc(xy, rad, col):
            r0, c0 = L.cell(xy)
            k = int(np.ceil(rad / L.res))
            for dr in range(-k, k + 1):
                for dc in range(-k, k + 1):
                    if dr * dr + dc * dc <= k * k and 0 <= r0 + dr < L.H and 0 <= c0 + dc < L.W:
                        img[r0 + dr, c0 + dc] = col
        for x, y, rad in L.floor:
            disc((x, y), rad, (150, 20, 20))
        for t, f in zip(self.tgt, self.found):
            disc(t[:2], 0.12, (40, 170, 60) if f else (230, 140, 0))
        if self.body_pts is None:
            disc(self.xy, self.cfg["robot"]["radius"], (30, 90, 220))
        else:
            c, s_ = np.cos(self.yaw), np.sin(self.yaw)
            for p in self.xy[None] + self.body_pts @ np.array([[c, s_], [-s_, c]]):
                r_, c_ = L.cell(p)
                img[r_, c_] = (30, 90, 220)
        disc(self.xy + 0.3 * np.array([np.cos(self.yaw), np.sin(self.yaw)]), 0.06, (255, 255, 255))
        return np.repeat(np.repeat(img[::-1], s, 0), s, 1)
