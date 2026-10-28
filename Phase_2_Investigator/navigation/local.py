"""Local control: pure pursuit along the global path + a LiDAR safety layer.

Safety layer (independent of the map, so it also protects against anything the
map got wrong):
- obstacle points = the current planar scan + low 3D-cloud points (z 0.04-0.7 m)
  from the last second, in the robot frame;
- forward clearance = distance from the front of the body to the nearest point
  inside the body's corridor (half width + margin); speed scales down from
  `slow_dist` and is zero below `stop_dist`; the same for reversing;
- turning on the spot is refused when a point is inside the body's sweep circle
  (its corners would hit), and the robot backs off first.

Go2 body (standing): 0.70 x 0.31 m, so half length 0.35, half width 0.155,
corner radius 0.38 m from the base centre.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from navigation.velocity import NAV_LIMITS


@dataclass
class SafetyConfig:
    half_length: float = 0.35
    half_width: float = 0.155
    corridor_margin: float = 0.03
    stop_dist: float = 0.20          # clearance (m) at which forward motion stops
    slow_dist: float = 0.60          # clearance at which slowing starts
    sweep_radius: float = 0.40       # turning in place needs this much room around the centre
    cloud_memory_s: float = 1.0


class SafetyLayer:
    def __init__(self, cfg: SafetyConfig = SafetyConfig()):
        self.cfg = cfg
        self.cloud = deque()             # (t, points_world (N,3))
        self.last = {"front": np.inf, "rear": np.inf, "nearest": np.inf, "limited": False}
        self.turn_blocked = False

    def observe(self, scan):
        pw = scan.points_world()
        band = pw[(pw[:, 2] > 0.04) & (pw[:, 2] < 0.70)]
        self.cloud.append((scan.t, band))
        while self.cloud and scan.t - self.cloud[0][0] > self.cfg.cloud_memory_s:
            self.cloud.popleft()
        self.scan = scan

    def _points_robot(self, state, extra=None):
        pts = [self.scan.scan_points_world()[:, :2]] + [c[:, :2] for _, c in self.cloud]
        if extra is not None and len(extra):
            pts.append(np.asarray(extra)[:, :2])
        p = np.vstack([x for x in pts if len(x)]) if any(len(x) for x in pts) else np.zeros((0, 2))
        c, s = np.cos(state.yaw), np.sin(state.yaw)
        d = p - state.position[:2]
        return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])

    def filter(self, state, vx, vy, wz, map_points=None):
        """Clip a (vx, vy, wz) command so it cannot drive into an obstacle.

        map_points: obstacle xy from the occupancy grid near the robot (covers low
        objects the LiDAR can no longer see up close)."""
        cfg = self.cfg
        p = self._points_robot(state, map_points)
        hw = cfg.half_width + cfg.corridor_margin
        inside = np.abs(p[:, 1]) < hw
        front = p[inside & (p[:, 0] > 0), 0] - cfg.half_length
        rear = -p[inside & (p[:, 0] < 0), 0] - cfg.half_length
        f = float(front.min()) if len(front) else np.inf
        r = float(rear.min()) if len(rear) else np.inf
        nearest = float(np.min(np.linalg.norm(p, axis=1))) if len(p) else np.inf
        limited = False

        def scale(clear):
            return float(np.clip((clear - cfg.stop_dist) / (cfg.slow_dist - cfg.stop_dist), 0.0, 1.0))

        if vx > 0 and scale(f) < 1:
            vx *= scale(f)
            limited = True
        if vx < 0 and scale(r) < 1:
            vx *= scale(r)
            limited = True
        if vy != 0:
            side = p[(np.abs(p[:, 0]) < cfg.half_length + 0.05) & (np.sign(p[:, 1]) == np.sign(vy)), 1]
            sclear = float(np.min(np.abs(side))) - cfg.half_width if len(side) else np.inf
            if sclear < cfg.stop_dist:
                vy, limited = 0.0, True
        if abs(wz) > 1e-3 and nearest < cfg.sweep_radius and self.motion_hits(p, vx, wz, 0.3, vy=vy):
            wz, limited = 0.0, True
            self.turn_blocked = True
        else:
            self.turn_blocked = False
        self.last = {"front": f, "rear": r, "nearest": nearest, "limited": limited}
        return vx, vy, wz


    @staticmethod
    def _moved(points, vx, vy, wz, tau):
        """Points expressed in the body frame after driving (vx, vy, wz) for tau seconds."""
        th = wz * tau
        if abs(wz) > 1e-6:
            dx = (vx * np.sin(th) - vy * (1 - np.cos(th))) / wz
            dy = (vx * (1 - np.cos(th)) + vy * np.sin(th)) / wz
        else:
            dx, dy = vx * tau, vy * tau
        c, s = np.cos(-th), np.sin(-th)
        d = points - np.array([dx, dy])
        return np.column_stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]])

    def motion_hits(self, p, vx, wz, horizon=0.5, margin=0.03, vy=0.0):
        """Would driving (vx, vy, wz) for `horizon` s put the body footprint on a point? (robot-frame points)"""
        cfg = self.cfg
        hl, hw = cfg.half_length + margin, cfg.half_width + margin
        reach = cfg.sweep_radius + (abs(vx) + abs(vy)) * horizon + 0.1
        near = p[np.linalg.norm(p, axis=1) < reach]
        if not len(near):
            return False
        for frac in (0.25, 0.5, 0.75, 1.0):
            q = self._moved(near, vx, vy, wz, horizon * frac)
            if np.any((np.abs(q[:, 0]) < hl) & (np.abs(q[:, 1]) < hw)):
                return True
        return False

    def _footprint_clearance(self, q):
        """Distance from the body rectangle to the nearest point (q in body frame)."""
        cfg = self.cfg
        dx = np.maximum(np.abs(q[:, 0]) - cfg.half_length, 0)
        dy = np.maximum(np.abs(q[:, 1]) - cfg.half_width, 0)
        return float(np.min(np.hypot(dx, dy))) if len(q) else 9.0

    def free_maneuver(self, state, desired_turn, map_points=None, horizon=0.8):
        """(vx, vy, wz) for a short collision-free move that best frees the robot.

        Candidates: turns, arcs, straight and sideways steps. Each is scored by the
        body's clearance after the move, plus a bonus for turning toward
        `desired_turn` (sign of the heading error). None if nothing is free."""
        p = self._points_robot(state, map_points)
        near = p[np.linalg.norm(p, axis=1) < 1.5]
        s = 1.0 if desired_turn >= 0 else -1.0
        cands = [(0.0, 0.0, 0.4 * s), (-0.12, 0.0, 0.4 * s), (0.12, 0.0, 0.4 * s), (-0.12, 0.0, -0.4 * s),
                 (0.12, 0.0, -0.4 * s), (-0.15, 0.0, 0.0), (0.15, 0.0, 0.0), (0.0, 0.12, 0.0), (0.0, -0.12, 0.0),
                 (0.0, 0.12, 0.3 * s), (0.0, -0.12, 0.3 * s)]
        best = None
        for vx, vy, wz in cands:
            if self.motion_hits(near, vx, wz, horizon, vy=vy):
                continue
            score = self._footprint_clearance(self._moved(near, vx, vy, wz, horizon)) + 0.05 * (wz * s > 0)
            if best is None or score > best[0]:
                best = (score, (vx, vy, wz))
        return None if best is None else best[1]


class PurePursuit:
    def __init__(self, lookahead=0.45, v_max=NAV_LIMITS.vx, w_max=NAV_LIMITS.wz, goal_tol=0.15,
                 turn_in_place=0.9):
        self.lookahead, self.v_max, self.w_max = lookahead, v_max, w_max
        self.goal_tol, self.turn_in_place = goal_tol, turn_in_place

    def command(self, state, path):
        """(vx, wz, done) to follow `path` (N, 2) from the robot's pose."""
        pos = state.position[:2]
        if path is None or len(path) == 0:
            return 0.0, 0.0, True
        d_goal = np.linalg.norm(path[-1] - pos)
        if d_goal < self.goal_tol:
            return 0.0, 0.0, True
        i0 = int(np.argmin(np.linalg.norm(path - pos, axis=1)))
        ahead = path[i0:]
        dist = np.linalg.norm(ahead - pos, axis=1)
        far = np.flatnonzero(dist >= self.lookahead)
        target = ahead[far[0]] if len(far) else path[-1]
        dx, dy = target - pos
        err = float(np.arctan2(dy, dx) - state.yaw)
        err = (err + np.pi) % (2 * np.pi) - np.pi
        if abs(err) > self.turn_in_place:
            return 0.0, float(np.sign(err) * self.w_max), False
        vx = self.v_max * max(0.2, np.cos(err)) * min(1.0, d_goal / 0.4 + 0.3)
        wz = float(np.clip(2.0 * err, -self.w_max, self.w_max))
        return float(vx), wz, False


def turn_towards(state, target_xy, w_max=NAV_LIMITS.wz, tol=0.06):
    """(wz, done) to face a point."""
    dx, dy = np.asarray(target_xy) - state.position[:2]
    err = (np.arctan2(dy, dx) - state.yaw + np.pi) % (2 * np.pi) - np.pi
    if abs(err) < tol:
        return 0.0, True
    return float(np.clip(1.5 * err, -w_max, w_max)), False
