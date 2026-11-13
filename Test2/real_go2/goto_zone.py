#!/usr/bin/env python3
"""Autonomous "go to zone" on the real Go2, driven by the trained zone classifier (real-world Q1, step 5).

    python real_go2/goto_zone.py --source sdk --iface eth1                    # then press 1-4 to pick a zone
    python real_go2/goto_zone.py --source sdk --iface eth1 --target shelves   # start straight away
    python real_go2/goto_zone.py --source sdk --iface eth1 --lidar-check      # no autonomy: check the LiDAR first

With the zone map (results/real_q1/zone_map.npz from `python real_go2/zone_map.py`, on by default when it
exists and the connection gives odometry):
  * the localiser runs all the time (also while you drive by hand): odometry + every classifier output ->
    where the dog is on the map. The mini-map (top right) says "map: here <zone>" once it is sure;
  * localised -> ROUTE: follow the shortest walkable path (the ground covered while recording) to the middle
    of the target zone, re-planning every few seconds, with the same LiDAR slow/steer/stop as a leg;
  * ARRIVED needs the map belief (navigation.map.arrive_prob) as well as the camera, so a zone the camera
    mistakes for the target is not accepted; if the camera says "here" but the map is unsure, it explores.
  Not localised yet: the classifier-only search below, while the localiser gathers evidence on the way.

How it finds a zone (camera classifier) and avoids obstacles on the way (LiDAR):
  LOOK      turn once on the spot, classifying every frame and storing the LiDAR free distance per heading
            (Test 1's body corridor, floor ignored). Out of one turn come:
            * where am I: the zone most of the views belong to;
            * which way: heading score = P(target) x free-space factor x blocked-memory penalty, so a
              direction with a desk in the way loses even if the target is behind it;
            * if the target's way is blocked: DETOUR to the open heading (>= want_free_m) closest to the
              target, staying on the same side; if the target is not seen at all: EXPLORE open space
              (momentum + obstacle-edge bonus), at most explore_legs; nothing open: fail "boxed in".
            If the target zone holds >= navigation.arrive_share of the views -> ARRIVED (waves hello).
  TURN      turn to the chosen heading.
  WALK      walk forward for one leg (navigation.leg_s). Every LiDAR scan: within slow_obstacle_m ahead ->
            slow down and steer to the freer side; within stop_obstacle_m -> stop, remember that heading
            as blocked, LOOK again. The leg also ends when the target is predicted steadily (early_stop_*)
            or fades (lost_*). Gives up after max_legs. Closed-loop check: real_go2/nav_sim.py.
Safety: the dog moves only through Test 1's teleop autonomy hook (go2_stream/teleop.py set_autonomy):
requests lapse unless refreshed, are clamped to the teleop limits, and ANY manual key (SPACE, W/A/S/D/Q/E,
ESC) cancels the mission at once. Walking needs LiDAR (Ethernet / --source sdk), as Test 1's missions do;
over Wi-Fi it only runs with --allow-no-lidar, slower and with shorter legs.
Keys (window): 1..N go to that zone, C cancel, G speed, V start/stop the MP4, ESC quit.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))
from real_go2.live_zone import COLORS, PANEL_W, VIEW_H, VIEW_W, Classifier, put_text  # noqa: E402
from real_go2.sources import GO2_STREAM, add_source_args, cycle_speed, open_link, start_teleop  # noqa: E402
from real_go2.zone_map import ZoneLocalizer, ZoneMap, lookahead_point, map_cfg  # noqa: E402

log = logging.getLogger("goto_zone")


def _go2_stream_first():
    """Make `perception` / `behavior` resolve to Test 1's go2_stream packages. Loading the Q1 model imports
    go2sim.paths, which puts ../Phase_2_Investigator (it has its own `perception` package) first on sys.path."""
    p = str(GO2_STREAM)
    if sys.path[0] != p:
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    for name in ("perception", "behavior"):
        mod = sys.modules.get(name)
        if mod is not None and not str(getattr(mod, "__file__", "") or "").startswith(p):
            for k in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
                del sys.modules[k]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def free_profile(points, nb, corridor, lookahead):
    """Free walking distance (m) in nb directions around the dog, from LiDAR points in the body frame
    (x forward, y left, z up). Direction i is i * 360/nb degrees counter-clockwise from straight ahead.
    For each direction the points are rotated into that direction's frame and Test 1's walking corridor is
    applied (half_width_m, z_min..z_max, from x_min); the free distance is the min_points-th nearest
    obstacle in it, or `lookahead` when the corridor is clear that far."""
    out = np.full(nb, float(lookahead))
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(1) & (p[:, 2] >= corridor["z_min"]) & (p[:, 2] <= corridor["z_max"])]
    if not len(p):
        return out
    th = np.arange(nb) * 2 * math.pi / nb
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    xr = c * p[None, :, 0] + s * p[None, :, 1]                # coordinates in each direction's frame
    yr = -s * p[None, :, 0] + c * p[None, :, 1]
    k = max(int(corridor["min_points"]), 1)
    for i in range(nb):
        m = (xr[i] >= corridor["x_min"]) & (xr[i] <= lookahead) & (np.abs(yr[i]) <= corridor["half_width_m"])
        if m.sum() >= k:
            out[i] = float(np.sort(xr[i][m])[k - 1])
    return out


class ZoneNavigator:
    """The decision logic, independent of the robot: feed it classifier outputs, the heading and (with LiDAR)
    the free-space profile; it returns velocity requests (vx, wz). States: idle, look, turn, walk, arrived,
    failed. Directions are absolute heading bins: bin b points at b * 360/heading_bins degrees.

    Obstacle avoidance is part of the path choice:
      * while looking around, every LiDAR profile is stored per absolute direction (the forward +-30 deg,
        where the sensor sees best), so a full turn maps the free distance in all directions;
      * the walking direction maximises P(target in that direction) x free-space factor, minus directions
        that blocked a recent leg; if the target's direction is blocked it takes a detour through the free
        direction closest to it;
      * while walking, an obstacle within slow_obstacle_m slows the dog and steers it to the freer side;
        within stop_obstacle_m the leg ends, that direction is remembered as blocked, and it plans again."""

    def __init__(self, classes, nav, have_lidar, zmap=None):
        self.classes, self.nav, self.have_lidar = list(classes), dict(nav), have_lidar
        # zone map (real_go2/zone_map.py): localisation from odometry + classifier, routes, map-checked arrival
        self.mcfg, self.zmap = map_cfg(nav), zmap
        self.loc = ZoneLocalizer(zmap, self.mcfg) if zmap is not None else None
        self.path, self.map_p, self._route_t0, self._replan_t = None, None, 0.0, 0.0
        if not have_lidar:
            self.nav["approach_speed"] = min(self.nav["approach_speed"], 0.15)
            self.nav["leg_s"] = min(self.nav["leg_s"], 2.0)
        self.nb = int(self.nav["heading_bins"])
        self.state, self.target, self.msg = "idle", None, "press 1-%d to choose a zone" % len(classes)
        self.legs = 0
        self.scan = []                      # (heading, prob vector) during LOOK
        self.best_heading, self.where, self.share = None, None, 0.0
        self.bins = self.free = self.score = None
        self.blocked = {}                   # absolute bin -> leg number when a leg ended blocked there
        self.detour = False
        self.exploring, self.explored = False, 0   # walking into open space because the target is not in view
        self._last_leg_heading = None
        self.detour_side = 0                # +1 going round obstacles on the left, -1 on the right, 0 none
        self._explore_heading = None
        self._world_free = np.full(self.nb, np.nan)
        self._recent = deque()              # (t, prob of target, target is argmax) during WALK
        self._t0 = 0.0
        self.avoiding = 0                   # -1 right, 0 none, +1 left (for the panel)
        self._last_free = (None, 0.0)       # latest LiDAR profile while walking (the LiDAR is slower than the loop)
        self._turn_t0 = None                # when the current TURN started (for turn_timeout_s)

    # ------------------------------------------------------------------ helpers
    def bin_of(self, heading):
        return int(round((heading % (2 * math.pi)) / (2 * math.pi) * self.nb)) % self.nb

    def heading_of(self, b):
        return b * 2 * math.pi / self.nb

    def _free_factor(self, free):
        nav = self.nav
        f = (free - nav["stop_obstacle_m"]) / max(nav["want_free_m"] - nav["stop_obstacle_m"], 1e-6)
        return np.where(np.isnan(free), 0.0, np.clip(f, 0.0, 1.0))

    # ------------------------------------------------------------------ commands
    def start(self, k, now, heading):
        self.target, self.legs, self.blocked = int(k), 0, {}
        self.explored, self.exploring, self._last_leg_heading = 0, False, None
        self.detour_side, self._explore_heading = 0, None
        self.path = None
        if self.zmap is not None:
            self.zmap.clear_blocks()
        self.msg = f"going to {self.classes[k]}"
        self._look(now, heading)

    def cancel(self, why="cancelled"):
        if self.state not in ("idle", "arrived", "failed"):
            self.state, self.msg = "idle", why

    @property
    def active(self):
        return self.state in ("look", "turn", "walk", "route")

    @property
    def map_status(self):
        """(localised?, confidence, zone index the belief favours, its probability) or None without a map."""
        if self.loc is None or self.loc.p0 is None:
            return None
        e, zb = self.loc.estimate(), self.loc.zone_belief()
        return self.loc.localized, e["conf"], int(zb.argmax()), float(zb.max())

    # ------------------------------------------------------------------ states
    def _look(self, now, heading):
        self.state, self.scan, self._t0, self._turn_t0 = "look", [], now, None
        self._belief_before_look = (float(self.loc.zone_belief()[self.target])
                                    if self.loc is not None and self.loc.p0 is not None and self.target is not None
                                    else 0.0)
        self._world_free = np.full(self.nb, np.nan)
        self.avoiding = 0
        self._last_free = (None, 0.0)

    def _store_free(self, heading, free):
        """Body-frame profile -> absolute directions, keeping the forward +-2 bins (the LiDAR's best view)."""
        cb = self.bin_of(heading)
        for j in (-2, -1, 0, 1, 2):
            self._world_free[(cb + j) % self.nb] = free[j % self.nb]

    def _finish_look(self, now, heading):
        nav, k, nb = self.nav, self.target, self.nb
        probs = np.array([p for _, p in self.scan])
        b = np.array([self.bin_of(h) for h, _ in self.scan])
        votes = np.bincount(probs.argmax(1), minlength=len(self.classes)) / len(probs)
        self.where, self.share = int(votes.argmax()), float(votes[k])
        sums, cnt = np.bincount(b, probs[:, k], nb), np.bincount(b, minlength=nb)
        mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        filled = np.where(np.isnan(mean), np.nanmean(mean), mean)
        p = (np.roll(filled, 1) + filled + np.roll(filled, -1)) / 3            # circular, +-1 bin
        self.bins = p
        tb = int(np.argmax(p))                                                  # where the target looks likeliest
        camera_here = self.share >= nav["arrive_share"]
        doubt = False                                                           # camera says "here", map disagrees
        if self.loc is not None and self.loc.p0 is not None:
            m = self.mcfg
            self.map_p = float(self.loc.zone_belief()[k])
            # zone certain, spot not yet: the belief was already that sure BEFORE this look-around (so a camera
            # that is wrong for one whole turn cannot talk the map into it on its own)
            sure_zone = (camera_here and self.map_p >= m["arrive_prob_unlocalized"]
                         and self._belief_before_look >= m["arrive_prob_unlocalized"])
            if sure_zone or (self.loc.localized and self.map_p >= m["arrive_prob"]
                             and self.share >= m["arrive_min_share"]):
                self.state = "arrived"
                self.msg = (f"ARRIVED at {self.classes[k]} (map {self.map_p:.0%}, camera {self.share:.0%} of the "
                            f"views)")
                return
            doubt = camera_here
        elif camera_here:
            self.state = "arrived"
            self.msg = f"ARRIVED at {self.classes[k]} ({self.share:.0%} of the views)"
            return
        if self.legs >= nav["max_legs"]:
            self.state = "failed"
            self.msg = f"gave up after {self.legs} legs (here looks like {self.classes[self.where]})"
            return
        if self.loc is not None and self.loc.localized and self._start_route(now):
            return
        self.exploring = False
        if doubt and not self.have_lidar:
            self.state = "failed"
            self.msg = (f"camera says {self.classes[k]} ({self.share:.0%}) but the map is unsure "
                        f"({self.map_p:.0%}); drive around a bit so it can localise")
            return
        if p[tb] < nav["seen_conf"] or doubt:
            # with the zone map, exploring is how the dog localises: allowed until max_legs
            budget = nav["explore_legs"] if self.loc is None else nav["max_legs"]
            if not (self.have_lidar and self.explored < budget):
                self.state = "failed"
                self.msg = f"{self.classes[k]} not seen in any direction (best {p[tb]:.2f})"
                return
            # explore: walk into the most open direction, not back the way it came, then look again
            self.free = self._world_free.copy()
            fac = self._free_factor(self.free)
            back = np.ones(nb)
            if self._last_leg_heading is not None:
                bb = self.bin_of(self._last_leg_heading + math.pi)
                for j in range(-3, 4):
                    back[(bb + j) % nb] = 0.3
            keep = np.ones(nb)                                                 # momentum: keep exploring the same way
            if self._explore_heading is not None:
                eb = self.bin_of(self._explore_heading)
                for j in range(-3, 4):
                    keep[(eb + j) % nb] = 1.6
            fz = np.nan_to_num(self.free, nan=0.0)
            edge = np.ones(nb)                                                 # obstacle edges lead round obstacles
            jump = np.maximum(np.abs(fz - np.roll(fz, 1)), np.abs(fz - np.roll(fz, -1)))
            edge[(jump >= nav["edge_jump_m"]) & (fz >= nav["want_free_m"])] = 1.8
            score = fz * (fac > 0) * back * keep * edge
            best = int(np.argmax(score))
            if score[best] <= 0:
                self.state = "failed"
                self.msg = f"{self.classes[k]} not seen and no free direction to explore"
                return
            self.exploring, self.explored, self.detour, self.score = True, self.explored + 1, False, score
            self.best_heading = self._explore_heading = self.heading_of(best)
            self.state = "turn"
            why = (f"camera says {self.classes[k]} but the map is not sure yet ({self.map_p:.0%})" if doubt
                   else f"{self.classes[k]} not in view (best {p[tb]:.2f})")
            self.msg = (f"{why}; exploring the most open way ({self.free[best]:.1f} m free, try "
                        f"{self.explored}/{budget})")
            return
        if self.have_lidar:
            self.free = self._world_free.copy()
            fac = self._free_factor(self.free)
            pen = np.ones(nb)
            for bb, leg in self.blocked.items():                               # avoid recently blocked ways
                if self.legs - leg < nav["blocked_memory_legs"]:
                    for j in (-1, 0, 1):
                        pen[(bb + j) % nb] *= 0.2
            self.score = p * fac * pen
            best = int(np.argmax(self.score))
            if self.score[best] < 0.25 * nav["seen_conf"]:                     # target's way is blocked: detour
                ok = np.flatnonzero((fac >= 0.6) & (pen >= 1.0))
                off = (ok - tb + nb / 2) % nb - nb / 2                          # signed bins from the target's way
                if self.detour_side:                                            # keep going round the same side
                    same = off * self.detour_side > 0
                    if same.any():
                        ok, off = ok[same], off[same]
                if not len(ok):
                    self.state = "failed"
                    self.msg = "boxed in: no free direction (LiDAR)"
                    return
                i = int(np.argmin(np.abs(off)))
                best = int(ok[i])
                self.detour_side = 1 if off[i] > 0 else -1
            else:
                self.detour_side = 0
        else:
            self.free, self.score, best = None, p, tb
        self.detour = abs((best - tb + nb / 2) % nb - nb / 2) > 1
        self.best_heading = self.heading_of(best)
        way = f"detour {math.degrees(wrap(self.best_heading - self.heading_of(tb))):+.0f} deg off the target's way" \
            if self.detour else "clear"
        free_txt = f", free {self.free[best]:.1f} m" if self.free is not None else ""
        self.state = "turn"
        self.msg = (f"here looks like {self.classes[self.where]}; heading {math.degrees(wrap(self.best_heading)):+.0f} "
                    f"deg towards {self.classes[k]} ({way}{free_txt})")

    def _start_route(self, now):
        """Localised on the zone map: plan a walkable route to the core of the target zone. False if none."""
        e = self.loc.estimate()
        path = self.zmap.plan(e["xy"], self.target, self.mcfg["core_prob"])
        if not path or len(path) < 2:
            return False
        self.path, self._route_t0, self._replan_t = path, now, now
        self.state, self.legs, self._last_free = "route", self.legs + 1, (None, 0.0)
        self.detour = self.exploring = False
        self.msg = (f"map: here is {self.classes[self.zmap.zone_at(e['xy'])]}; route to {self.classes[self.target]}"
                    f" {self._path_left(e['xy']):.1f} m (leg {self.legs})")
        return True

    def _path_left(self, xy):
        return lookahead_point(self.path, xy, 0.0)[1] if self.path else 0.0

    def _avoid(self, now, heading, free, vx, wz):
        """LiDAR while walking: (vx, wz, reason to end the leg or None). Within slow_obstacle_m ahead: slow down and
        steer to the freer side; within stop_obstacle_m: stop and remember the direction as blocked."""
        nav = self.nav
        self.avoiding = 0
        if free is not None:
            self._last_free = (free, now)
        elif self._last_free[0] is not None and now - self._last_free[1] <= 0.5:
            free = self._last_free[0]                                         # hold the last scan between scans
        if not (self.have_lidar and free is not None):
            return vx, wz, None
        free = np.asarray(free, float)
        if not np.any(free > 0):
            return 0.0, 0.0, "no LiDAR data"
        front = float(free[0])
        react = nav.get("reaction_s", 0.0) * max(vx, 0.0)                    # faster -> stop and slow earlier
        if front <= nav["stop_obstacle_m"] + react:
            self.blocked[self.bin_of(heading)] = self.legs
            return 0.0, 0.0, f"blocked {front:.2f} m ahead"
        if front <= nav["slow_obstacle_m"] + react:                            # steer round it, slowly
            left, right = min(free[1], free[2]), min(free[-1], free[-2])
            self.avoiding = 1 if left >= right else -1
            return nav["approach_speed"] * 0.4, self.avoiding * nav["avoid_turn_rate"], None
        return vx, wz, None

    def _route_step(self, now, heading, free):
        nav, m, k = self.nav, self.mcfg, self.target
        e = self.loc.estimate()
        end, vx, wz = None, 0.0, 0.0
        if e is None or e["conf"] < m["lost_conf"]:
            end = f"map position uncertain ({0 if e is None else e['conf']:.0%})"
        else:
            if now - self._replan_t >= m["replan_s"]:                          # the estimate keeps improving
                self._replan_t = now
                p = self.zmap.plan(e["xy"], k, m["core_prob"])
                if p:
                    self.path = p
            pt, left = lookahead_point(self.path, e["xy"], m["lookahead_m"])
            if left < 0.4 and np.hypot(*(pt - e["xy"])) < 0.6:
                end = f"reached the planned spot in {self.classes[k]}"
            elif now - self._route_t0 >= m["route_max_s"]:
                end = "route leg time limit"
            else:
                want = self.loc.to_odom_heading(math.atan2(pt[1] - e["xy"][1], pt[0] - e["xy"][0]))
                self.best_heading = want
                err = wrap(want - heading)
                if abs(err) > math.radians(m["turn_in_place_deg"]):              # far off: stop and turn first
                    rate = min(abs(nav.get("turn_gain", 1.5) * err), nav.get("turn_max_rate", nav["scan_rate"]))
                    wz = math.copysign(max(rate, nav.get("turn_min_rate", 0.25)), err)
                else:
                    vx, wz = nav["approach_speed"], float(np.clip(1.2 * err, -0.6, 0.6))
                    vx, wz, end = self._avoid(now, heading, free, vx, wz)
                    if end and end.startswith("blocked"):                       # plan round it next time
                        self.zmap.block(e["xy"] + 0.6 * np.array([math.cos(e["heading"]), math.sin(e["heading"])]))
                self.msg = (f"route to {self.classes[k]}: {left:.1f} m to go (map {e['conf']:.0%} sure)")
        if end is not None:
            self.msg = f"{end}; looking around"
            self._look(now, heading)
            return 0.0, 0.0
        return vx, wz

    def update(self, now, heading, prob=None, free=None, pos=None):
        """One control step. prob: latest classifier output (None if no new frame). free: latest LiDAR free-space
        profile in the body frame (free_profile; None if no new scan; all zeros = no LiDAR data). pos: odometry
        (x, y) for the zone map's localiser (it runs in every state, also while driving by hand).
        Returns (vx, wz)."""
        nav, k = self.nav, self.target
        if self.loc is not None and pos is not None:
            if prob is not None:
                self.loc.update(pos[:2], heading, prob, now)
            else:
                self.loc.move(pos[:2], heading)
        if self.state == "route":
            return self._route_step(now, heading, free)
        if self.state == "look":
            if prob is not None:
                self.scan.append((heading, np.asarray(prob, float)))
            if free is not None and self.have_lidar:
                self._store_free(heading, np.asarray(free, float))
            turned = sum(abs(wrap(b[0] - a[0])) for a, b in zip(self.scan, self.scan[1:]))
            if turned >= 2 * math.pi - 0.05 and len(self.scan) >= 8:
                self._finish_look(now, heading)
                return 0.0, 0.0
            return 0.0, nav["scan_rate"]
        if self.state == "turn":
            err = wrap(self.best_heading - heading)
            if self._turn_t0 is None:
                self._turn_t0 = now
            timed_out = now - self._turn_t0 >= nav.get("turn_timeout_s", 12.0)
            if timed_out and abs(err) > math.radians(20):
                self._turn_t0 = None
                self.msg = f"turn did not reach the heading ({math.degrees(err):+.0f} deg left); looking around"
                self._look(now, heading)
                return 0.0, 0.0
            if abs(err) < nav.get("turn_tolerance_rad", 0.10) or timed_out:
                self._turn_t0 = None
                self.state, self.legs, self._t0, self._recent = "walk", self.legs + 1, now, deque()
                self._last_leg_heading = self.best_heading
                kind = ", exploring" if self.exploring else ", detour" if self.detour else ""
                self.msg = f"walking towards {self.classes[k]} (leg {self.legs}{kind})"
                return 0.0, 0.0
            # proportional, but never below turn_min_rate: the Go2 barely turns on the spot at small rates
            rate = min(abs(nav.get("turn_gain", 1.5) * err), nav.get("turn_max_rate", nav["scan_rate"]))
            return 0.0, math.copysign(max(rate, nav.get("turn_min_rate", 0.25)), err)
        if self.state == "walk":
            if prob is not None:
                self._recent.append((now, float(prob[k]), int(np.argmax(prob)) == k))
            while self._recent and now - self._recent[0][0] > max(nav["early_stop_s"], nav["lost_s"]):
                self._recent.popleft()
            wz = float(np.clip(0.8 * wrap(self.best_heading - heading), -0.3, 0.3))
            vx, wz, end = self._avoid(now, heading, free, nav["approach_speed"], wz)
            if (self.loc is not None and end is None and now - self._replan_t >= 1.0 and self.loc.localized):
                self._replan_t = now
                if self._start_route(now):                                     # localised mid-leg: switch to the map
                    return 0.0, 0.0
            span = now - self._recent[0][0] if self._recent else 0.0
            # with the map, the camera alone does not cut a leg short (it stopped legs after 2 s in a loop on the
            # real dog): only when the map also puts the dog in the target zone
            camera_may_stop = self.loc is None or float(self.loc.zone_belief()[k]) >= self.mcfg["arrive_prob"]
            if end is None and not self.detour and camera_may_stop and span >= nav["early_stop_s"] * 0.9:
                w = [r for r in self._recent if now - r[0] <= nav["early_stop_s"]]
                if all(r[2] for r in w) and np.mean([r[1] for r in w]) >= nav["early_stop_conf"]:
                    end = f"{self.classes[k]} recognised steadily"
            if end is None and not (self.detour or self.exploring) and span >= nav["lost_s"] * 0.9:
                w = [r for r in self._recent if now - r[0] <= nav["lost_s"]]
                if max(r[1] for r in w) < nav["lost_conf"]:
                    end = f"lost sight of {self.classes[k]}"
            leg_s = nav["leg_s"]
            if self.exploring and self.loc is not None and not self.loc.localized:
                leg_s = self.mcfg["explore_leg_s"]                              # not localised: cover more ground
            if end is None and now - self._t0 >= leg_s:
                end = "leg done"
            if end is not None:
                self.msg = f"{end}; looking around"
                self._look(now, heading)
                return 0.0, 0.0
            return vx, wz
        return 0.0, 0.0


class Heading:
    """Heading for the navigator: the dog's IMU yaw when available (Ethernet), else the integral of the
    commanded turn rate (Wi-Fi / fake: approximate)."""

    def __init__(self, pose_fn=None):
        self.pose_fn, self.h, self.t = pose_fn, 0.0, None

    def __call__(self, now, wz_cmd):
        if self.pose_fn is not None:
            p = self.pose_fn()
            if p is not None:
                return float(p[2])
        if self.t is not None:
            self.h = wrap(self.h + wz_cmd * (now - self.t))
        self.t = now
        return self.h


_MAP_BASE = {}


def map_overlay(view, nav, classes, size=190):
    """Top-right of the camera view: the zone map (colour = zone, dark = never walked), the dog's estimated
    position and heading (white when localised, grey while unsure), the planned route, and the belief."""
    zm = nav.zmap
    if zm is None:
        return
    nx, ny = zm.shape
    s = size / max(nx, ny)
    if id(zm) not in _MAP_BASE:
        img = np.full((ny, nx, 3), 40, np.uint8)
        k = zm.prob.argmax(2)
        for z in range(len(classes)):
            m = (zm.walk & (k == z)).T[::-1]
            img[m] = (0.55 * np.array(COLORS[z % len(COLORS)]) + 0.45 * 90).astype(np.uint8)
        _MAP_BASE[id(zm)] = cv2.resize(img, (int(nx * s), int(ny * s)), interpolation=cv2.INTER_NEAREST)
    img = _MAP_BASE[id(zm)].copy()
    h = img.shape[0]

    def px(xy):
        return int((xy[0] - zm.lo[0]) / zm.res * s), int(h - (xy[1] - zm.lo[1]) / zm.res * s)

    if nav.path is not None and nav.state == "route":
        cv2.polylines(img, [np.array([px(q) for q in nav.path], np.int32)], False, (255, 255, 255), 1)
    st = nav.map_status
    if st is not None:
        ok, conf, zb, zp = st
        e = nav.loc.estimate()
        c = px(e["xy"])
        col = (255, 255, 255) if ok else (150, 150, 150)
        cv2.circle(img, c, 4, col, -1)
        cv2.line(img, c, (int(c[0] + 10 * math.cos(e["heading"])), int(c[1] - 10 * math.sin(e["heading"]))), col, 2)
        txt = (f"map: here {classes[zb]} {zp:.0%} (sure {conf:.0%})" if ok else
               f"map: localising {conf:.0%} - drive around")
    else:
        txt = "map: waiting for odometry"
    x0 = view.shape[1] - img.shape[1] - 8
    view[8:8 + h, x0:x0 + img.shape[1]] = img
    cv2.rectangle(view, (x0 - 1, 7), (x0 + img.shape[1], 8 + h), (200, 200, 200), 1)
    put_text(view, txt, (max(x0 - 120, 4), 8 + h + 18), 0.45)


def nav_panel(nav, classes, prob, fps, recording):
    p = np.full((VIEW_H, PANEL_W, 3), 24, np.uint8)
    put_text(p, "Go to zone - real Go2", (12, 26), 0.6, (255, 255, 255))
    put_text(p, "ResNet18 zone classifier + LiDAR safety", (12, 46), 0.42, (170, 170, 170))
    tgt = classes[nav.target] if nav.target is not None else "-"
    put_text(p, f"target: {tgt}", (12, 80), 0.75, COLORS[nav.target % len(COLORS)] if nav.target is not None
             else (200, 200, 200), 2)
    put_text(p, f"state: {nav.state.upper()}   legs {nav.legs}", (12, 108), 0.5)
    words, line, y = nav.msg.split(), "", 130
    for w in words:                                          # wrap the status message
        if len(line + " " + w) > 40:
            put_text(p, line, (12, y), 0.42, (200, 200, 200))
            line, y = w, y + 18
        else:
            line = (line + " " + w).strip()
    put_text(p, line, (12, y), 0.42, (200, 200, 200))
    k = int(np.argmax(prob))
    put_text(p, f"camera sees: {classes[k]} ({prob[k]:.2f})", (12, 186), 0.5)
    if nav.avoiding:
        put_text(p, f"avoiding obstacle: steering {'left' if nav.avoiding > 0 else 'right'}", (12, 206), 0.45,
                 (0, 200, 255))
    if nav.bins is not None and nav.target is not None:      # last look-around, per absolute direction
        cx, cy, r = PANEL_W // 2, 305, 70
        cv2.circle(p, (cx, cy), r, (70, 70, 70), 1)
        nb = len(nav.bins)
        look = nav.nav.get("lookahead_m", 2.5)
        for i in range(nb):
            a = i / nb * 2 * math.pi                         # up = heading 0, counter-clockwise
            ux, uy = -math.sin(a), -math.cos(a)
            if nav.free is not None and not np.isnan(nav.free[i]):   # grey: free distance
                rf = int(r * min(nav.free[i] / look, 1.0))
                cv2.line(p, (cx, cy), (int(cx + rf * ux), int(cy + rf * uy)), (95, 95, 95), 7)
            rr = int(r * float(np.clip(nav.bins[i], 0, 1)))  # colour: target likelihood
            cv2.line(p, (cx, cy), (int(cx + rr * ux), int(cy + rr * uy)), COLORS[nav.target % len(COLORS)], 3)
        if nav.best_heading is not None:                     # white: chosen direction
            a = nav.best_heading
            cv2.line(p, (cx, cy), (int(cx - (r + 8) * math.sin(a)), int(cy - (r + 8) * math.cos(a))),
                     (255, 255, 255), 1)
        put_text(p, "look-around: target (colour), free space (grey), chosen (white)", (12, 226), 0.36,
                 (170, 170, 170))
        if nav.where is not None:
            put_text(p, f"here looked like {classes[nav.where]}; target share {nav.share:.0%}", (12, 392), 0.42)
    put_text(p, f"{fps:4.1f} fps   1-{len(classes)} go  C cancel  V video", (12, VIEW_H - 30), 0.42, (170, 170, 170))
    put_text(p, "REC" if recording else "not recording (V)", (12, VIEW_H - 10), 0.45,
             (60, 60, 235) if recording else (150, 150, 150))
    return p


def lidar_panel(free, nav_cfg, fps):
    """--lidar-check: the live LiDAR free distance per direction around the dog (up = straight ahead) and what
    the navigator would do with it. No mission and no autonomous motion."""
    p = np.full((VIEW_H, PANEL_W, 3), 24, np.uint8)
    put_text(p, "LiDAR check - no autonomy", (12, 26), 0.6, (255, 255, 255))
    put_text(p, "free distance in Test 1's body corridor", (12, 46), 0.42, (170, 170, 170))
    if free is None:
        put_text(p, "waiting for the LiDAR ...", (12, 90), 0.55)
        return p
    look = nav_cfg["lookahead_m"]
    ahead = float(free[0])
    stop, slow = nav_cfg["stop_obstacle_m"], nav_cfg["slow_obstacle_m"]
    if not free.any():
        verdict, col = "NO LiDAR DATA (would not walk)", (60, 60, 235)
    elif ahead <= stop:
        verdict, col = f"would STOP (< {stop:.2f} m)", (60, 60, 235)
    elif ahead <= slow:
        verdict, col = f"would SLOW + steer (< {slow:.2f} m)", (0, 200, 255)
    else:
        verdict, col = "clear ahead", (90, 230, 90)
    put_text(p, f"ahead: {ahead:.2f} m", (12, 82), 0.7, (255, 255, 255), 2)
    put_text(p, verdict, (12, 108), 0.5, col)
    put_text(p, f"left {free[len(free) // 4]:.2f} m   right {free[-len(free) // 4]:.2f} m", (12, 132), 0.48)
    cx, cy, r = PANEL_W // 2, 270, 110
    for rad, c in ((stop, (60, 60, 235)), (slow, (0, 200, 255)), (look, (90, 90, 90))):
        cv2.circle(p, (cx, cy), int(r * min(rad / look, 1.0)), c, 1)
    nb = len(free)
    for i in range(nb):
        a = i / nb * 2 * math.pi
        rf = int(r * min(free[i] / look, 1.0))
        cv2.line(p, (cx, cy), (int(cx - rf * math.sin(a)), int(cy - rf * math.cos(a))),
                 (170, 170, 170) if i else (255, 255, 255), 3 if i else 5)
    put_text(p, "rings: stop (red), slow (orange), lookahead", (12, VIEW_H - 48), 0.4, (170, 170, 170))
    put_text(p, f"{fps:4.1f} fps   drive with teleop keys, ESC quit", (12, VIEW_H - 28), 0.42, (170, 170, 170))
    return p


class LidarProfile:
    """Free-space profile from the real LiDAR (Ethernet): Test 1's LidarSource + cloud_to_body conversion,
    then free_profile with Test 1's walking corridor. Returns a new profile per new cloud (None otherwise);
    all zeros when the cloud is missing or older than max_age_s (= blocked: no data, no walking)."""

    def __init__(self, link, go2cfg, nav, max_age_s=1.0):
        import lidar as lidar_mod
        bcfg = yaml.safe_load((Path(lidar_mod.__file__).parent / "config/behaviors.yaml").read_text())
        self.corridor = dict(bcfg["safety"]["corridor"], **(nav.get("corridor") or {}))
        self.nav, self.max_age_s = nav, max_age_s
        self.lidar_mod, self.state = lidar_mod, link.state
        lcfg = dict(go2cfg["lidar"])
        lcfg["topic"] = nav.get("lidar_topic", lcfg["topic"])    # Test 1's config file itself is not changed
        self.src = lidar_mod.LidarSource(lcfg)
        self._seq, self._warned = -1, False

    def __call__(self):
        from perception.pipeline import cloud_to_body
        msg, seq, stamp = self.src.latest.get()
        nb = int(self.nav["heading_bins"])
        if msg is None or time.monotonic() - stamp > self.max_age_s:
            return np.zeros(nb)
        if seq == self._seq:
            return None
        self._seq = seq
        frame_id = str(getattr(msg.header, "frame_id", ""))
        if "odom" not in frame_id.lower():
            # A cloud in the sensor's own frame (e.g. rt/utlidar/cloud, 'utlidar_lidar') is upside down and tilted
            # relative to the body: the floor lands inside the corridor. Treat it as no data (= no walking).
            if not self._warned:
                log.error("LiDAR cloud frame is '%s', not odom: free space can't be computed from it. Use "
                          "navigation.lidar_topic: rt/utlidar/cloud_deskewed in real_config.yaml.", frame_id)
                self._warned = True
            return np.zeros(nb)
        sport = self.state.sport.get()[0] if self.state is not None else None
        pts, _ = cloud_to_body(msg, sport, "auto", self.lidar_mod)
        if pts is None:
            return np.zeros(nb)
        return free_profile(pts, nb, self.corridor, self.nav["lookahead_m"])


def apply_nav_speed(nav, nav_cfg, name):
    """Go-to-zone's own walking / look-around speeds for a preset (navigation.speed_presets in real_config.yaml).
    Without the LiDAR the walking speed stays capped at 0.15 m/s."""
    p = nav_cfg.get("speed_presets", {}).get(name)
    if not p:
        return
    nav.nav["scan_rate"] = float(p["scan_rate"])
    nav.nav["approach_speed"] = float(p["approach_speed"]) if nav.have_lidar else min(float(p["approach_speed"]), 0.15)
    log.info("Go-to-zone speed %s: walk %.2f m/s, look-around %.2f rad/s", name.upper(),
             nav.nav["approach_speed"], nav.nav["scan_rate"])


def make_safety(link, go2cfg, nav, allow_no_lidar):
    """(free-space profile fn or None, pose fn or None). Ethernet: real LiDAR profile + odometry heading."""
    if link.name == "sdk":
        _go2_stream_first()
        from behavior.clearance import make_real_pose_fn
        from perception.pipeline import cloud_to_body  # noqa: F401  (load now, before anything can shadow it)
        return LidarProfile(link, go2cfg, nav), make_real_pose_fn(link.state)
    if not allow_no_lidar:
        raise SystemExit("Autonomous walking needs the LiDAR (Ethernet, --source sdk) for obstacle avoidance. Over "
                         "Wi-Fi, run with --allow-no-lidar only in a clear area with someone ready to press SPACE.")
    log.warning("No LiDAR: obstacle avoidance OFF, speed capped at 0.15 m/s, 2 s legs. Supervise closely.")
    if link.name == "webrtc":                       # Wi-Fi still has the dog's IMU heading: turns use it
        def pose():
            p = link.pose()
            return None if math.isnan(p[2]) else p
        return None, pose
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--config", default=str(TEST2 / "real_go2/real_config.yaml"))
    ap.add_argument("--model", help="checkpoint (default: <results_dir>/resnet18/best.pt)")
    ap.add_argument("--target", help="zone to go to at start (name or 1-based number)")
    ap.add_argument("--allow-no-lidar", action="store_true", help="Wi-Fi / fake: walk without the obstacle check")
    ap.add_argument("--map", help="zone map (default: navigation.map.file, made by real_go2/zone_map.py)")
    ap.add_argument("--no-map", action="store_true", help="classifier-only navigation (no localisation / routes)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--lidar-check", action="store_true",
                    help="no autonomy: show the live LiDAR free space and whether the navigator would stop/slow "
                         "(tune Test 1's safety.corridor with this before the first mission)")
    ap.add_argument("--seconds", type=float, help="stop after this many seconds")
    ap.add_argument("--video-fps", type=float, default=10.0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(a.config).read_text())
    out = TEST2 / cfg["results_dir"]
    out.mkdir(parents=True, exist_ok=True)
    clf = Classifier(a.model or out / "resnet18/best.pt", cfg["crop"])
    classes = clf.classes
    link, go2cfg = open_link(a)
    a.no_control = False
    tele = start_teleop(link, go2cfg, a, keyboard=not a.headless)
    profile, pose_fn = make_safety(link, go2cfg, cfg["navigation"], a.allow_no_lidar)
    zmap = None
    if not a.no_map:
        mpath = Path(a.map) if a.map else TEST2 / map_cfg(cfg["navigation"])["file"]
        if pose_fn is None:
            log.warning("No odometry on this connection: zone map off (classifier-only navigation).")
        elif not mpath.exists():
            log.warning("No zone map at %s: classifier-only navigation. Build it with  python real_go2/zone_map.py",
                        mpath)
        else:
            zmap = ZoneMap.load(mpath)
            if zmap.classes != list(classes):
                log.warning("Zone map classes %s != model classes %s: map off.", zmap.classes, classes)
                zmap = None
            else:
                log.info("Zone map %s: localising from odometry + camera. Drive around for a minute (different "
                         "zones) until the map says 'here ...', or let the missions explore.", mpath.name)
    nav = ZoneNavigator(classes, cfg["navigation"], have_lidar=profile is not None, zmap=zmap)
    apply_nav_speed(nav, cfg["navigation"], tele.speed_name)
    heading = Heading(pose_fn)
    tele.on_manual_input = lambda key: nav.cancel(f"cancelled by key '{key}'")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4 = out / f"goto_{stamp}.mp4"
    events, missions = [], []          # missions: one entry per zone request (target, result, reason, legs, time)
    map_trace, next_trace = [], 0.0    # zone map belief every 0.5 s: t, odometry x y yaw, map x y, sure, zone, P
    writer, recording = None, a.headless
    prob, wz, last_seq, last_state = np.ones(len(classes)) / len(classes), 0.0, -1, "idle"
    t0 = time.monotonic()
    next_video, last_free, last_print = t0, None, 0.0
    if a.lidar_check and profile is None:
        tele.shutdown()
        link.close()
        raise SystemExit("--lidar-check needs the LiDAR: use --source sdk --iface <name> (Ethernet).")

    def begin(k, now, h):
        if missions and missions[-1]["result"] is None:          # a new request replaces an unfinished one
            missions[-1].update(result="replaced", reason="new zone requested", legs=nav.legs,
                                seconds=round(now - t0 - missions[-1]["t"], 1))
        nav.start(k, now, h)
        missions.append({"target": classes[k], "t": round(now - t0, 2), "result": None, "speed": tele.speed_name})

    if a.target and not a.lidar_check:
        k = classes.index(a.target) if a.target in classes else int(a.target) - 1
        begin(k, t0, heading(t0, 0.0))
    try:
        while True:
            now = time.monotonic()
            h = heading(now, wz)
            img, seq = link.frame()
            new = None
            if img is not None and seq != last_seq:
                last_seq = seq
                prob = new = clf(img)
            if a.lidar_check:
                f = profile() if profile is not None else None
                if f is not None:
                    last_free = f
                    if now - last_print >= 1.0:
                        log.info("LiDAR free ahead %.2f m, left %.2f m, right %.2f m", f[0], f[len(f) // 4],
                                 f[-len(f) // 4])
                        last_print = now
                view = np.zeros((VIEW_H, VIEW_W, 3), np.uint8) if img is None else cv2.resize(img, (VIEW_W, VIEW_H))
                comp = np.hstack([view, lidar_panel(last_free, cfg["navigation"], link.fps())])
                if a.seconds and now - t0 >= a.seconds:
                    break
                if a.headless:
                    time.sleep(0.02)
                    continue
                cv2.imshow("Go2 LiDAR check", comp)
                if (cv2.waitKey(1) & 0xFF) == 27 or tele.quit_event.is_set():
                    break
                continue
            free = profile() if (profile is not None and nav.active) else None
            pose = pose_fn() if (pose_fn is not None and zmap is not None) else None
            vx, wz = nav.update(now, h, new, free, pos=None if pose is None else pose[:2])
            if pose is not None and now >= next_trace and nav.map_status is not None:
                next_trace = now + 0.5
                ok, conf, zb, zp = nav.map_status
                e = nav.loc.estimate()
                map_trace.append([round(now - t0, 2), *(round(float(v), 3) for v in pose),
                                  round(float(e["xy"][0]), 2), round(float(e["xy"][1]), 2), round(conf, 3),
                                  classes[zb], round(zp, 3)])
            if nav.active:
                tele.set_autonomy(vx, 0.0, wz)
            else:
                tele.clear_autonomy()
                wz = 0.0
            if nav.state != last_state:
                events.append({"t": round(now - t0, 2), "state": nav.state, "msg": nav.msg})
                log.info("%s: %s", nav.state, nav.msg)
                if nav.state in ("arrived", "failed", "idle") and missions and missions[-1]["result"] is None:
                    missions[-1].update(result="cancelled" if nav.state == "idle" else nav.state, reason=nav.msg, legs=nav.legs,
                                        explored=getattr(nav, "explored", 0),
                                        map=None if nav.map_status is None else
                                        {"localised": bool(nav.map_status[0]), "sure": round(nav.map_status[1], 3),
                                         "zone": classes[nav.map_status[2]], "p": round(nav.map_status[3], 3)},
                                        seconds=round(now - t0 - missions[-1]["t"], 1))
                if nav.state == "arrived" and cfg["navigation"].get("hello_on_arrival"):
                    tele.request_action("hello")
                last_state = nav.state
            view = np.zeros((VIEW_H, VIEW_W, 3), np.uint8) if img is None else cv2.resize(img, (VIEW_W, VIEW_H))
            put_text(view, f"speed {tele.speed_name.upper()} (G)", (12, VIEW_H - 14), 0.5)
            map_overlay(view, nav, classes)
            comp = np.hstack([view, nav_panel(nav, classes, prob, link.fps(), recording)])
            if recording:
                if writer is None:
                    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), a.video_fps,
                                             (comp.shape[1], comp.shape[0]))
                while now >= next_video:
                    writer.write(comp)
                    next_video += 1.0 / a.video_fps
            else:
                next_video = now
            if a.seconds and now - t0 >= a.seconds:
                break
            if a.headless:
                time.sleep(0.02)
                continue
            cv2.imshow("Go2 go to zone", comp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or tele.quit_event.is_set():
                break
            if ord("1") <= key < ord("1") + len(classes):
                begin(key - ord("1"), now, h)
            elif key in (ord("c"), ord("C")):
                nav.cancel()
            elif key in (ord("g"), ord("G")):
                name = cycle_speed(tele, go2cfg)
                apply_nav_speed(nav, cfg["navigation"], name)
                events.append({"t": round(now - t0, 2), "state": nav.state, "msg": f"speed -> {name}"})
            elif key in (ord("v"), ord("V")):
                recording = not recording
    except KeyboardInterrupt:
        pass
    finally:
        tele.clear_autonomy()
        tele.shutdown()
        link.close()
        if writer is not None:
            writer.release()
        if not a.headless:
            cv2.destroyAllWindows()
        if missions and missions[-1]["result"] is None:
            missions[-1].update(result="unfinished", reason="program ended", legs=nav.legs)
        (out / f"goto_{stamp}.json").write_text(json.dumps(
            {"classes": classes, "target": classes[nav.target] if nav.target is not None else None,
             "final_state": nav.state, "legs": nav.legs, "source": link.name, "lidar": profile is not None,
             "video": mp4.name if writer is not None else None, "zone_map": zmap is not None, "missions": missions,
             "events": events, "map_trace_columns": ["t", "odom_x", "odom_y", "odom_yaw", "map_x", "map_y", "sure",
                                                     "zone", "p"], "map_trace": map_trace},
            indent=1))
        print(f"{nav.state}: {nav.msg}")


if __name__ == "__main__":
    main()
