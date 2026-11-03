"""Waypoint follower: drives the Go2 through the crime scene with velocity commands only.

Each goal is reached by
1. planning a collision-free path on the inflated ground-truth costmap (`StaticMap.plan`, Phase 2's
   planner), then
2. pure pursuit on that path: a look-ahead point gives the heading error e;
   yaw_rate = k_yaw * e (clipped), vx = speed * max(0, cos e) and zero while |e| is large
   (turn on the spot first), a small vy corrects the cross-track error.
Optional "look-around" spins at a goal turn the dog on the spot so the camera sees the zone from
all sides (good for the Q1 classifier). Everything goes through `CrimeSceneGo2.set_velocity_command`.

A callback `on_tick(dog)` runs every control tick (CONTROL_DT = 0.02 s); the data recorder uses it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from go2sim.go2_env import CONTROL_DT


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class FollowerConfig:
    lookahead: float = 0.35
    k_yaw: float = 1.6
    k_lat: float = 0.6
    turn_in_place_err: float = 0.7     # rad: above this, stop and turn first
    goal_tol: float = 0.18             # m
    stuck_window_s: float = 6.0        # no 0.1 m progress in this time -> give up on the goal
    replan_dev: float = 0.5            # m off the path -> replan


class WaypointFollower:
    def __init__(self, dog, static_map, cfg: FollowerConfig = FollowerConfig(), on_tick=None):
        self.dog, self.map, self.cfg = dog, static_map, cfg
        self.on_tick = on_tick or (lambda dog: None)
        self.path_log = []          # (t, x, y) of every tick, for plots
        self.events = []            # (t, kind, info)

    def _tick(self, vx, vy, wz):
        self.dog.set_velocity_command(vx, vy, wz)
        self.dog.step(CONTROL_DT)
        x, y, _ = self.dog.pose2d()
        self.path_log.append((self.dog.time, x, y))
        self.on_tick(self.dog)

    def hold(self, seconds):
        for _ in range(int(round(seconds / CONTROL_DT))):
            self._tick(0.0, 0.0, 0.0)

    def spin(self, angle, rate=0.45, min_clear=0.42):
        """Turn on the spot by `angle` radians (signed). Skipped when the body would sweep into an
        obstacle (the Go2's corners are 0.38 m from the base centre)."""
        if self.map.clearance_at(self.dog.pose2d()[:2]) < min_clear:
            self.events.append((self.dog.time, "spin_skipped", 0.0))
            return False
        yaw0 = self.dog.pose2d()[2]
        turned, last = 0.0, yaw0
        sign = np.sign(angle) or 1.0
        t0 = self.dog.time
        while abs(turned) < abs(angle) - 0.05 and self.dog.time - t0 < abs(angle) / rate * 2.5 + 2:
            self._tick(0.0, 0.0, sign * rate)
            yaw = self.dog.pose2d()[2]
            turned += wrap(yaw - last)
            last = yaw
        self.events.append((self.dog.time, "spin", float(turned)))
        return True

    def go_to(self, goal_xy, speed=0.3, max_time=90.0):
        """Walk to goal_xy. Returns True when reached."""
        cfg = self.cfg
        goal_xy = np.asarray(goal_xy, float)
        start = self.dog.pose2d()
        path = self.map.plan(start[:2], goal_xy)
        if path is None:
            self.events.append((self.dog.time, "unreachable", goal_xy.tolist()))
            return False
        goal_eff = path[-1]
        t0 = self.dog.time
        best, best_t = np.inf, t0
        last_replan = -np.inf
        while self.dog.time - t0 < max_time:
            x, y, yaw = self.dog.pose2d()
            p = np.array([x, y])
            dist_goal = np.linalg.norm(goal_eff - p)
            if dist_goal < cfg.goal_tol:
                self.events.append((self.dog.time, "reached", goal_eff.tolist()))
                return True
            if dist_goal < best - 0.1:
                best, best_t = dist_goal, self.dog.time
            elif self.dog.time - best_t > cfg.stuck_window_s:
                self.events.append((self.dog.time, "stuck", goal_eff.tolist()))
                self.hold(0.2)
                return False
            d = np.linalg.norm(path - p, axis=1)
            i = int(np.argmin(d))
            if d[i] > cfg.replan_dev and self.dog.time - last_replan > 1.0:
                # off the path: replan from here (at most once a second; always step afterwards,
                # so a path that starts away from the dog cannot stall the simulation)
                last_replan = self.dog.time
                new = self.map.plan(p, goal_xy)
                if new is not None:
                    path, goal_eff = new, new[-1]
                    d = np.linalg.norm(path - p, axis=1)
                    i = int(np.argmin(d))
            # look-ahead point
            j = i
            while j < len(path) - 1 and np.linalg.norm(path[j] - p) < cfg.lookahead:
                j += 1
            target = path[j]
            dx, dy = target - p
            err = wrap(np.arctan2(dy, dx) - yaw)
            wz = np.clip(cfg.k_yaw * err, -0.5, 0.5)
            if abs(err) > cfg.turn_in_place_err:
                vx = vy = 0.0
            else:
                slow = min(1.0, dist_goal / 0.5 + 0.3)
                vx = speed * max(0.0, np.cos(err)) * slow
                # cross-track error of the closest path segment, body-frame lateral correction
                a = path[max(i - 1, 0)]
                b = path[min(i + 1, len(path) - 1)]
                seg = b - a
                if np.linalg.norm(seg) > 1e-6:
                    n = np.array([-seg[1], seg[0]]) / np.linalg.norm(seg)
                    cross = float((p - a) @ n)
                    lat_world = -cfg.k_lat * cross * n
                    vy = float(-np.sin(yaw) * lat_world[0] + np.cos(yaw) * lat_world[1])
                else:
                    vy = 0.0
            self._tick(vx, vy, wz)
        self.events.append((self.dog.time, "timeout", goal_eff.tolist()))
        return False
