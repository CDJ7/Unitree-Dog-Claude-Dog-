"""Autonomous navigation on the simulated Go2: map, plan, follow, explore, log collisions.

    nav = Navigator(sim)
    report = nav.explore(max_time=400)          # frontier exploration until no frontiers remain
    nav.go_to((x, y))                           # plan + drive to a point

Control loop at 10 Hz (simulation time): LiDAR -> occupancy grid + safety
layer; global path replanned every 1.5 s (or when blocked); pure pursuit ->
safety filter -> cmd_vel (navigation/velocity.py, 0.3 m/s, 0.5 rad/s). If the
robot makes no progress for `stuck_s`, it backs off and the goal is
blacklisted. Every contact between the robot and anything but the floor is
logged as a collision (none are expected).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from navigation.frontier import choose_frontier
from navigation.local import PurePursuit, SafetyLayer
from navigation.occupancy import OccupancyGrid
from navigation.planner import CostField, nearest_reachable, simplify
from navigation.velocity import NAV_LIMITS, SimVelocity

DT = 0.1
ROBOT_RADIUS = 0.32          # preferred planning inflation (m): keeps the robot in open space
TIGHT_RADIUS = 0.24          # fallback when nothing else connects: 0.48 m gaps are passable


@dataclass
class CollisionEvent:
    t: float
    robot_geom: str
    other: str
    position: list


@dataclass
class NavLog:
    path: list = field(default_factory=list)                 # [(t, x, y, yaw)]
    collisions: list = field(default_factory=list)
    goals: list = field(default_factory=list)                # [(t, kind, x, y, result)]
    safety_stops: int = 0


class Navigator:
    def __init__(self, sim, grid=None, robot_radius=ROBOT_RADIUS, on_tick=None, verbose=True):
        self.sim = sim
        self.vel = SimVelocity(sim, NAV_LIMITS)
        self.safety = SafetyLayer()
        self.follower = PurePursuit()
        st = sim.state()
        if grid is None:
            grid = OccupancyGrid(origin=st.position[:2] - np.array([7.0, 5.0]), size=(14.0, 10.0))
        self.grid = grid
        self.grid.clear_disk(st.position[:2], 0.3)
        self.robot_radius = robot_radius
        self.log = NavLog()
        self.on_tick = list(on_tick or [])
        self.verbose = verbose
        self._contacts = set()
        from navigation.coverage import CameraCoverage
        self.coverage = CameraCoverage(self.grid)
        self.blacklist = []
        self._dist = None
        self.scan = None
        self.tick(0, 0, 0)

    # ------------------------------------------------------------ basic loop
    def say(self, *a):
        if self.verbose:
            print(f"[nav t={self.sim.time:6.1f}s]", *a, flush=True)

    def tick(self, vx, vy, wz, safe=True):
        """Sense, filter the command, act for DT seconds."""
        self.scan = self.sim.lidar()
        self.grid.update(self.scan)
        self.safety.observe(self.scan)
        st = self.sim.state()
        self._k = getattr(self, "_k", 0) + 1
        if self._k % 2 == 0:                    # depth camera at 5 Hz: low obstacles close in front
            self.grid.add_low_points(self.sim.depth_points(), st.position[:2])
        if safe:
            near = self.grid.occupied_points_near(st.position[:2], 1.2)
            vx2, vy2, wz2 = self.safety.filter(st, vx, vy, wz, map_points=near)
            if (vx2, vy2, wz2) != (vx, vy, wz) and (abs(vx) > 0.05 or abs(wz) > 0.05):
                self.log.safety_stops += int(abs(vx2) < 1e-6 and abs(vx) > 0.05)
            vx, vy, wz = vx2, vy2, wz2
        if abs(vx) + abs(vy) + abs(wz) < 1e-6:
            self.vel.stop()
        else:
            self.vel.send(vx, vy, wz)
        self.sim.step(DT)
        self._check_contacts()
        self.coverage.update(self.sim)
        st = self.sim.state()
        self.log.path.append((st.t, float(st.position[0]), float(st.position[1]), float(st.yaw)))
        for cb in self.on_tick:
            cb(self)
        return st

    def _check_contacts(self):
        now = set()
        for c in self.sim.contacts():
            other = c.other_geom if c.other_body == "world" else c.other_body
            key = (c.robot_geom, other)
            now.add(key)
            if key not in self._contacts:
                self.log.collisions.append(CollisionEvent(c.t, c.robot_geom, other,
                                                          np.round(c.position, 3).tolist()))
                self.say(f"COLLISION {c.robot_geom} <-> {other} at {np.round(c.position, 2).tolist()}")
        self._contacts = now

    def hold(self, seconds):
        for _ in range(int(round(seconds / DT))):
            self.tick(0, 0, 0)

    # ----------------------------------------------------------- go to point
    def field(self, radius=None):
        cost = self.grid.costmap(robot_radius=radius or self.robot_radius)
        return CostField(self.grid, cost, self.sim.state().position[:2])

    def _clear_at(self, xy):
        r, c = self.grid.to_cell(xy)
        if self._dist is None or not self.grid.inside(r, c):
            return 9.0
        return float(self._dist[r, c])

    def plan(self, goal_xy, tol=0.6):
        """Path to goal_xy: through open space if possible, else through tight gaps."""
        self._dist = self.grid.obstacle_distance()
        for radius in (self.robot_radius, TIGHT_RADIUS):
            f = self.field(radius)
            rc = nearest_reachable(f, goal_xy, max_radius=tol)
            if rc is not None:
                return simplify(f.path_to(rc))
        return None

    def go_to(self, goal_xy, timeout=90.0, tol=0.6, replan_s=1.5, stuck_s=8.0, final_tol=0.15):
        """Drive to goal_xy (or the nearest reachable point within tol). True on arrival."""
        goal_xy = np.asarray(goal_xy, float)
        t0 = self.sim.time
        path = self.plan(goal_xy, tol)
        if path is None:
            self.say(f"no path to {goal_xy.round(2).tolist()}")
            return False
        self.follower.goal_tol = final_tol
        last_plan = self.sim.time
        best, best_t = np.inf, self.sim.time
        blocked_since, recoveries = None, 0
        while self.sim.time - t0 < timeout:
            st = self.sim.state()
            d = float(np.linalg.norm(path[-1] - st.position[:2]))
            if d < best - 0.05:
                best, best_t = d, self.sim.time
            if self.safety.turn_blocked:
                blocked_since = blocked_since or self.sim.time
                if self.sim.time - blocked_since > 1.0 and recoveries < 8:
                    # cannot turn on the spot: a short collision-free arc (three-point turn)
                    tgt = path[min(len(path) - 1, 3)]
                    err = (np.arctan2(tgt[1] - st.position[1], tgt[0] - st.position[0]) - st.yaw + np.pi) \
                        % (2 * np.pi) - np.pi
                    near = self.grid.occupied_points_near(st.position[:2], 1.2)
                    man = self.safety.free_maneuver(st, err, near)
                    if man is not None:
                        for _ in range(8):
                            self.tick(*man)
                    recoveries += 1
                    blocked_since = None
                    best_t = self.sim.time
                    continue
            else:
                blocked_since = None
            if self.sim.time - best_t > stuck_s:
                sl = self.safety.last
                self.say(f"stuck at {np.round(st.position[:2], 2).tolist()} yaw {np.degrees(st.yaw):.0f} "
                         f"(goal {np.round(path[-1], 2).tolist()}, {d:.2f} m; front {sl['front']:.2f} "
                         f"rear {sl['rear']:.2f} nearest {sl['nearest']:.2f}, turn blocked "
                         f"{self.safety.turn_blocked}): backing off")
                self.back_off()
                return False
            if self.sim.time - last_plan > replan_s:
                p = self.plan(goal_xy, tol)
                if p is None:
                    self.say("goal became unreachable")
                    return False
                path, last_plan = p, self.sim.time
            vx, wz, done = self.follower.command(st, path)
            if self._clear_at(st.position[:2]) < 0.40:
                vx *= 0.5                       # tight spot: half speed for accurate tracking
            if not done and self.safety.last["limited"] and d < 0.45:
                done = True                     # as close as the safety layer allows
            if done:
                self.tick(0, 0, 0)
                return True
            self.tick(vx, 0, wz)
        self.say("go_to timed out")
        return False

    def back_off(self, distance=1.0, timeout=12.0):
        """Reverse out the way the robot came: steer the rear along its own recent path."""
        hist = np.array([(x, y) for _, x, y, _ in self.log.path[::-1]])
        if len(hist) < 2:
            return
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(hist, axis=0), axis=1))]
        trail = hist[(d > 0.05) & (d < distance)]
        t0 = self.sim.time
        for target in trail[::3]:
            while self.sim.time - t0 < timeout:
                st = self.sim.state()
                to = target - st.position[:2]
                if np.linalg.norm(to) < 0.08:
                    break
                back_yaw = st.yaw + np.pi
                err = (np.arctan2(to[1], to[0]) - back_yaw + np.pi) % (2 * np.pi) - np.pi
                self.tick(-0.12, 0, float(np.clip(1.5 * err, -0.4, 0.4)))
                if self.safety.last["rear"] < self.safety.cfg.stop_dist + 0.02:
                    break
        self.tick(0, 0, 0)

    def turn_to(self, target_xy, timeout=12.0):
        from navigation.local import turn_towards
        t0 = self.sim.time
        while self.sim.time - t0 < timeout:
            wz, done = turn_towards(self.sim.state(), target_xy)
            if done:
                self.tick(0, 0, 0)
                return True
            self.tick(0, 0, wz)
        return False

    # ------------------------------------------------------------- explore
    def explore(self, max_time=400.0, spin_first=True):
        """Frontier exploration until no reachable frontier remains or time runs out."""
        t0, wall0 = self.sim.time, time.time()
        if spin_first:                          # look around before the first plan
            for _ in range(int(2 * np.pi / NAV_LIMITS.wz / DT)):
                self.tick(0, 0, NAV_LIMITS.wz)
        n_goals = 0
        while self.sim.time - t0 < max_time:
            f = self.field()
            pick = choose_frontier(self.grid, f, self.blacklist, min_clear=0.50)
            if pick is None:
                self.say("no reachable frontiers left: exploration complete")
                break
            goal, gxy, _ = pick
            n_goals += 1
            self.say(f"frontier goal {n_goals}: {np.round(gxy, 2).tolist()}")
            ok = self.go_to(gxy, timeout=60.0, tol=0.3, final_tol=0.25)
            self.log.goals.append((self.sim.time, "frontier", float(gxy[0]), float(gxy[1]), bool(ok)))
            # never pick the same goal twice: if its frontier is still there after a visit,
            # it cannot be observed from the floor (under furniture, behind thin geometry)
            self.blacklist.append(gxy)
        self.tick(0, 0, 0)
        return self.report(t0, wall0)

    def patrol(self, max_time=300.0):
        from navigation.coverage import patrol
        return patrol(self, self.coverage, max_time=max_time)

    def report(self, t0=0.0, wall0=None):
        path = np.array([(x, y) for _, x, y, _ in self.log.path])
        length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))) if len(path) > 1 else 0.0
        return {
            "sim_time_s": round(self.sim.time - t0, 1),
            "wall_time_s": round(time.time() - wall0, 1) if wall0 else None,
            "path_length_m": round(length, 2),
            "known_free_m2": round(float(self.grid.free().sum()) * self.grid.res ** 2, 2),
            "camera_coverage": self.coverage.coverage(),
            "collisions": [vars(c) for c in self.log.collisions],
            "n_collisions": len(self.log.collisions),
            "safety_stops": self.log.safety_stops,
            "goals": self.log.goals,
        }
