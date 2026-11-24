"""Autonomous zone mapping: explore with the LiDAR, look around at each stop, tag the map with zones.

    mapper = ZoneMapper(nav, classifier)          # nav: navigation.explorer.Navigator (sim or real session)
    report = mapper.run(max_time=600)
    mapper.zone_map.save("outputs/zone_map/x")

Exploration is Phase 2's frontier exploration (navigation/explorer.py) with one change: at every
frontier goal the robot turns a full circle on the spot. The classifier sees each place from every
direction, which is what the zone map needs (one view shows where the dog faces, not where it stands).
While driving, frames are tagged too (every `every_s` seconds).
The LiDAR maps a room from a few spots, so frontiers run out long before the camera has seen every part
of it. A coverage phase then visits viewpoints spread ~`spacing` m apart over all reachable free space
(nearest first, by path) and looks around at each, until every reachable area has had a look-around
nearby. Zone evidence therefore covers the whole floor instead of a few places.
"""
from __future__ import annotations

import time

import numpy as np

from mapping.zone_map import ZoneMap
from navigation.frontier import choose_frontier
from navigation.velocity import NAV_LIMITS


class ZoneMapper:
    def __init__(self, nav, classifier, zone_map=None, every_s=0.5, look_rate=None, log=print, **zm_kw):
        self.nav, self.clf = nav, classifier
        self.sim = nav.sim
        self.zone_map = zone_map or ZoneMap(nav.grid, classifier.classes if classifier is not None else ["unknown"],
                                            **zm_kw)
        self.every_s, self.look_rate = every_s, look_rate or NAV_LIMITS.wz
        self.log = log
        self.next_t = -np.inf
        self.last_prob = None
        self.looks = []                                    # (x, y) of every look-around
        self.recent = []                                   # probabilities since the last look_around() began
        nav.on_tick.append(self)

    # ---------------------------------------------------------------- tagging
    def __call__(self, nav):
        t = self.sim.time
        if self.clf is None or t < self.next_t:            # no classifier: explore and look around only
            return
        self.next_t = t + self.every_s
        frame = self.sim.camera()
        if frame is None or frame.rgb is None:
            return
        st = self.sim.state()
        prob = self.clf(frame.rgb)
        self.zone_map.add(st.position[:2], st.yaw, prob)
        self.last_prob = prob
        self.recent.append(prob)

    def look_around(self):
        """Turn a full circle on the spot (the safety layer may hold the turn near obstacles)."""
        nav = self.nav
        st = nav.sim.state()
        self.looks.append(tuple(np.round(st.position[:2], 2)))
        self.recent = []
        turned, prev = 0.0, st.yaw
        t0 = nav.sim.time
        blocked = 0
        while turned < 2 * np.pi - 0.1 and nav.sim.time - t0 < 2.5 * 2 * np.pi / self.look_rate:
            st = nav.tick(0, 0, self.look_rate)
            turned += abs((st.yaw - prev + np.pi) % (2 * np.pi) - np.pi)
            prev = st.yaw
            blocked = blocked + 1 if nav.safety.turn_blocked else 0
            if blocked >= 10:                              # 1 s unable to turn here: step clear, then go on
                near = nav.grid.occupied_points_near(st.position[:2], 1.2)
                man = nav.safety.free_maneuver(st, +1, near)
                if man is not None:
                    for _ in range(8):
                        nav.tick(*man)
                    prev = nav.sim.state().yaw
                blocked = 0
        nav.tick(0, 0, 0)

    # --------------------------------------------------------------- the run
    def coverage_goal(self, spacing=1.0, min_clear=0.35):
        """The nearest reachable free point (by path) that is at least `spacing` m from every look-around so
        far, with `min_clear` m to obstacles (room to turn on the spot); or None when the floor is covered."""
        nav = self.nav
        g = nav.grid
        # reachable with the normal planning radius only: viewpoints that need the tight radius got the robot
        # wedged in a gap in simulation (r1854, 4 stuck goals in a row), so they are left out
        f = nav.field()
        clear = g.obstacle_distance()
        cand = g.free() & np.isfinite(f.cum) & (clear >= min_clear)
        ko = getattr(g, "keepout", None)
        if ko is not None and ko.any():                     # never stand (and spin) on floor evidence
            from scipy.ndimage import distance_transform_edt as _edt
            cand &= _edt(~ko) * g.res > 0.40
        rr, cc = np.nonzero(cand)
        if not len(rr):
            return None
        xy = g.to_xy(rr, cc)
        done = np.array(self.looks + [tuple(b) for b in nav.blacklist], float).reshape(-1, 2)
        if len(done):
            d = np.min(np.linalg.norm(xy[:, None, :] - done[None, :, :], axis=2), axis=1)
            keep = d >= spacing
            rr, cc, xy = rr[keep], cc[keep], xy[keep]
        if not len(rr):
            return None
        i = int(np.argmin(f.cum[rr, cc]))
        return xy[i]

    def run(self, max_time=900.0, fill_in=True, max_fill=40, spacing=1.0):
        nav = self.nav
        t0, wall0 = nav.sim.time, time.time()
        self.look_around()
        n = 0
        while nav.sim.time - t0 < max_time:
            pick = choose_frontier(nav.grid, nav.field(), nav.blacklist, min_clear=0.50)
            if pick is None:
                self.log("[map] no reachable frontiers left")
                break
            _, gxy, _ = pick
            n += 1
            self.log(f"[map t={nav.sim.time - t0:5.0f}s] frontier {n}: go to {np.round(gxy, 2).tolist()}")
            ok = nav.go_to(gxy, timeout=60.0, tol=0.3, final_tol=0.25)
            nav.log.goals.append((nav.sim.time, "frontier", float(gxy[0]), float(gxy[1]), bool(ok)))
            nav.blacklist.append(gxy)
            self.look_around()
            self.log(f"[map] {self.status()}")
        k = 0
        while fill_in and k < max_fill and nav.sim.time - t0 < max_time:
            gxy = self.coverage_goal(spacing)
            if gxy is None:
                self.log("[map] every reachable area has had a look-around")
                break
            k += 1
            self.log(f"[map t={nav.sim.time - t0:5.0f}s] viewpoint {k}: {np.round(gxy, 2).tolist()}")
            ok = nav.go_to(gxy, timeout=60.0, tol=0.3, final_tol=0.25)
            nav.log.goals.append((nav.sim.time, "viewpoint", float(gxy[0]), float(gxy[1]), bool(ok)))
            nav.blacklist.append(gxy)
            if ok:
                self.look_around()
        nav.tick(0, 0, 0)
        rep = nav.report(t0, wall0)
        rep.update({"frames_tagged": self.zone_map.frames, "look_arounds": len(self.looks),
                    "regions": self.zone_map.regions()})
        return rep

    def status(self):
        reg = self.zone_map.regions()
        return (f"{self.zone_map.frames} frames tagged; "
                + (", ".join(f"{k} {v['area_m2']:.1f} m2" for k, v in reg.items()) or "no zones yet"))
