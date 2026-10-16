"""Approach-and-act mission: walk up to a detected object, stop at a safe distance, perform a small action.

State machine, run on its own thread. It never sends robot commands itself: it hands velocity requests to
teleop.Teleop (whose single control thread owns Move(), the deadman and the watchdog) and asks it to run actions.

    IDLE -> READY -(go-ahead)-> APPROACH -> SETTLE -> [AWAIT_CONFIRM] -> ACTING -> HOLD -> RISING -> DONE
                         any problem or key press at any point -> ABORTED (robot is told to stop)

Safety properties (all unit-tested in the sim and with fakes; NOT yet run on the real robot):
  * needs an explicit go-ahead call after a target has been seen in `confirm_frames` consecutive results;
  * needs the dog to be standing (Teleop.standing, set by pressing R in this program);
  * any human control key (Teleop.on_manual_input) aborts at once, and Teleop also drops the velocity;
  * velocity requests expire after Teleop's deadman_s, so a dead mission thread stops the dog;
  * stale detections stop the dog, a target lost for target_lost_s aborts, as do timeouts and a blocked path;
  * speeds are capped here and again by Teleop's limits; never strafes or reverses; no Damp, no risky actions.
"""
import logging
import threading
import time
from enum import Enum
from typing import Callable, Optional

from .control import (ALLOWED_ACTIONS, Target, approach_command, candidates, pick_target, validate_config)

log = logging.getLogger("mission")


class State(Enum):
    IDLE = "idle"
    READY = "ready"
    APPROACH = "approach"
    SETTLE = "settle"
    AWAIT_CONFIRM = "await_confirm"
    ACTING = "acting"
    HOLD = "hold"
    RISING = "rising"
    DONE = "done"
    ABORTED = "aborted"


ACTIVE = (State.APPROACH, State.SETTLE, State.AWAIT_CONFIRM, State.ACTING, State.HOLD, State.RISING)


class Mission(threading.Thread):
    def __init__(self, cfg, get_result: Callable, teleop, classes=None, clearance_fn: Optional[Callable] = None,
                 dry_run=False, allow_autonomy=True, rate_hz=20):
        super().__init__(name="mission", daemon=True)
        self.cfg, self.p = cfg, cfg["safety"]
        self.table = validate_config(cfg)                         # {class: (action, stop_distance)}
        self.classes = [c for c in (classes or list(self.table)) if c in self.table]
        unknown = [c for c in (classes or []) if c not in self.table]
        if unknown:
            raise ValueError(f"no behaviour configured for {unknown}; configured: {list(self.table)}")
        self.get_result, self.teleop, self.clearance_fn = get_result, teleop, clearance_fn
        self.dry_run, self.allow_autonomy, self.period = dry_run, allow_autonomy, 1.0 / rate_hz
        self._lock = threading.RLock()
        self._stop_evt = threading.Event()
        self.state = State.READY
        self.message = ""
        self.candidate: Optional[Target] = None
        self.target: Optional[Target] = None
        self.stop_dist = 0.0
        self.action = ""
        self._seen = 0
        self._last_seq = -1
        self._last_target_seq = -1
        self._t_state = time.monotonic()
        self._t_start = 0.0
        self._last_seen = 0.0
        self._blocked_since: Optional[float] = None
        self._settle_since: Optional[float] = None
        self._confirmed = False
        self._act_t = 0.0
        self._hold_until = 0.0
        self._last_cmd_log = 0.0
        self.last_cmd = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ public API (UI / teleop threads)
    def go(self):
        """The go-ahead. Returns (started, message)."""
        with self._lock:
            if self.state in (State.DONE, State.ABORTED):
                self._set(State.READY, "re-armed: wait for a target, then press the go key again")
                return False, self.message
            if self.state != State.READY:
                return False, f"cannot start from state {self.state.value}"
            if not self.allow_autonomy:
                return False, "autonomy is disabled: start with --allow-autonomy (real robot) after reading config/behaviors.yaml"
            c = self.candidate
            if c is None or self._seen < self.p["confirm_frames"]:
                return False, "no confirmed target yet (need it in view for a moment)"
            if not self.teleop.standing:
                return False, "dog is not known to be standing: press R first"
            action, stop = self.table[c.cls]
            if c.range > self.p["max_start_distance_m"]:
                return False, f"{c.cls} is {c.range:.1f} m away; the limit is {self.p['max_start_distance_m']} m"
            if c.range < stop - self.p["stop_tolerance_m"]:
                return False, f"{c.cls} is already closer ({c.range:.2f} m) than the stop distance {stop:.2f} m"
            self.target, self.stop_dist, self.action = c, stop, action
            self._t_start = self._last_seen = time.monotonic()
            self._blocked_since = self._settle_since = None
            self._confirmed = False
            self._set(State.APPROACH, f"approaching {c.cls} ({c.range:.2f} m) -> {action} at {stop:.2f} m"
                                      + (" [DRY RUN: nothing will move]" if self.dry_run else ""))
            return True, self.message

    def confirm(self):
        with self._lock:
            if self.state == State.AWAIT_CONFIRM:
                self._confirmed = True

    def abort(self, reason):
        with self._lock:
            if self.state not in ACTIVE:
                return
            self._set(State.ABORTED, f"aborted: {reason}")
        self._zero()

    def manual_input(self, key):
        """Hook for Teleop.on_manual_input."""
        self.abort(f"manual input ({key})")

    def stop(self):
        self._stop_evt.set()
        self._zero()

    def status_text(self):
        with self._lock:
            s, c = self.state, self.candidate
            if s == State.READY:
                if not self.allow_autonomy:
                    return "MISSION: autonomy disabled (--allow-autonomy)"
                if c is None or self._seen < 1:
                    return f"MISSION: looking for {'/'.join(self.classes)}"
                if self._seen < self.p["confirm_frames"]:
                    return f"MISSION: {c.cls} {c.range:.2f} m (confirming)"
                act, stop = self.table[c.cls]
                stand = "" if self.teleop.standing else "  (press R first)"
                return f"MISSION READY: {c.cls} {c.range:.2f} m -> {act} @ {stop:.2f} m  [B = go]{stand}"
            if s == State.APPROACH and self.target:
                t = self.target
                return f"MISSION{' DRY' if self.dry_run else ''}: approaching {t.cls} {t.range:.2f} m -> {self.stop_dist:.2f} m  [any key aborts]"
            if s == State.AWAIT_CONFIRM:
                return f"MISSION: at the {self.target.cls}; press C to {self.action}"
            if s in (State.SETTLE, State.ACTING, State.HOLD, State.RISING):
                return f"MISSION: {s.value} ({self.action})"
            return f"MISSION {s.value}: {self.message}"

    # ------------------------------------------------------------------ internals
    def _set(self, state, message=""):
        self.state, self.message, self._t_state = state, message, time.monotonic()
        if message:
            log.info("%s: %s", state.value.upper(), message)

    def _zero(self):
        self.last_cmd = (0.0, 0.0, 0.0)
        if not self.dry_run:
            try:
                self.teleop.clear_autonomy()
            except Exception:
                log.exception("clear_autonomy failed")

    def _publish(self, vx, vyaw, now):
        self.last_cmd = (vx, 0.0, vyaw)
        if self.dry_run:
            if now - self._last_cmd_log > 0.5:
                self._last_cmd_log = now
                t = self.target
                log.info("[dry run] would command vx=%+.2f vyaw=%+.2f   (target %.2f m ahead %.2f m left, bearing %+.0f deg)",
                         vx, vyaw, t.x, t.y, 57.2958 * t.bearing)
        else:
            self.teleop.set_autonomy(vx, 0.0, vyaw)

    def run(self):
        while not self._stop_evt.is_set():
            t0 = time.monotonic()
            try:
                self._tick(t0)
            except Exception as e:
                log.exception("mission error")
                self.abort(f"internal error: {e}")
            time.sleep(max(self.period - (time.monotonic() - t0), 0.0))

    def _fresh_result(self):
        res = self.get_result()
        if res is None or time.time() - res.wall_time > self.p["stale_result_s"]:
            return None
        return res

    def _tick(self, now):
        with self._lock:
            s = self.state
            if s in ACTIVE and (self.teleop.quit_event.is_set() if hasattr(self.teleop, "quit_event") else False):
                self.abort("program is exiting")
                return
            if s == State.READY:
                self._tick_ready()
            elif s == State.APPROACH:
                self._tick_approach(now)
            elif s == State.SETTLE:
                self._tick_settle(now)
            elif s == State.AWAIT_CONFIRM:
                self._zero()
                if self._confirmed:
                    self._start_action(now)
                elif now - self._t_state > 20.0:
                    self._set(State.ABORTED, "aborted: no confirmation within 20 s")
            elif s == State.ACTING:
                if now - self._act_t > 0.4 and self.teleop.idle():
                    hold = self.cfg["actions"][self.action].get("hold_s", 0.0)
                    then = self.cfg["actions"][self.action].get("then", "none")
                    if then == "rise":
                        self._hold_until = now + hold
                        self._set(State.HOLD, f"holding {self.action} for {hold:.0f} s")
                    else:
                        self._set(State.DONE, f"{self.action} finished")
            elif s == State.HOLD:
                if now >= self._hold_until:
                    self.teleop.request_action("rise")
                    self._act_t = now
                    self._set(State.RISING, "rising")
            elif s == State.RISING:
                if now - self._act_t > 0.4 and self.teleop.idle():
                    self._set(State.DONE, f"{self.action} finished, standing again")

    def _tick_ready(self):
        res = self._fresh_result()
        if res is None or res.seq == self._last_seq:
            return
        self._last_seq = res.seq
        cands = candidates(res.detections, self.classes, self.p["min_conf"], res.wall_time)
        pick = pick_target(cands, self.candidate) if self.candidate is not None else None   # same object as before?
        if pick is not None:
            self._seen += 1
        else:                                       # new or lost: start counting again from the nearest candidate
            pick = pick_target(cands, None)
            self._seen = 1 if pick else 0
        self.candidate = pick

    def _update_target(self, res, now):
        if res is not None and res.seq != self._last_target_seq:
            self._last_target_seq = res.seq
            cands = candidates(res.detections, self.classes, self.p["min_conf"], res.wall_time)
            t = pick_target(cands, self.target)
            if t is not None:
                self.target, self._last_seen = t, now

    def _check_common(self, now):
        """Abort conditions shared by APPROACH and SETTLE. Returns True if the mission was aborted."""
        if not self.teleop.standing:
            self._set(State.ABORTED, "aborted: dog is no longer standing")
            self._zero()
            return True
        if now - self._last_seen > self.p["target_lost_s"]:
            self._set(State.ABORTED, "aborted: target lost")
            self._zero()
            return True
        return False

    def _tick_approach(self, now):
        if now - self._t_start > self.p["max_travel_s"]:
            self._set(State.ABORTED, "aborted: approach took too long")
            self._zero()
            return
        if self.dry_run and now - self._t_start > self.cfg["dry_run_report_s"]:
            self._set(State.DONE, "dry run finished: nothing was sent to the robot")
            return
        self._update_target(self._fresh_result(), now)
        if self._check_common(now):
            return
        if now - self._last_seen > self.p["stale_result_s"]:
            self._publish(0.0, 0.0, now)                 # no fresh sighting: stand still and wait (until lost)
            return
        t = self.target
        if self.clearance_fn is not None:
            d = self.clearance_fn((t.x, t.y))
            if d is not None and d < self.p["corridor"]["stop_if_closer_m"]:
                self._publish(0.0, 0.0, now)
                self._blocked_since = self._blocked_since or now
                self.message = f"path blocked {d:.2f} m ahead"
                if now - self._blocked_since > self.p["blocked_abort_s"]:
                    self._set(State.ABORTED, f"aborted: path blocked ({d:.2f} m ahead)")
                    self._zero()
                return
        self._blocked_since = None
        vx, vyaw, arrived = approach_command(t, self.stop_dist, self.p)
        if arrived and not self.dry_run:
            self._publish(0.0, 0.0, now)
            self._settle_since = now
            self._set(State.SETTLE, f"arrived ({t.range:.2f} m); settling")
        else:
            self._publish(vx, vyaw, now)

    def _tick_settle(self, now):
        self._publish(0.0, 0.0, now)
        self._update_target(self._fresh_result(), now)
        if self._check_common(now):
            return
        t = self.target
        if t.range > self.stop_dist + self.p["stop_tolerance_m"] + 0.15:     # it moved away: go after it again
            self._t_start = now
            self._set(State.APPROACH, "target moved; approaching again")
            return
        if now - self._settle_since >= self.p["settle_s"]:
            if self.p["confirm_before_action"]:
                self._confirmed = False
                self._set(State.AWAIT_CONFIRM, f"waiting for C to {self.action}")
            else:
                self._start_action(now)

    def _start_action(self, now):
        if self.action not in ALLOWED_ACTIONS:           # belt and braces: validate_config already enforces this
            self._set(State.ABORTED, f"aborted: action {self.action} not allowed")
            return
        self.teleop.request_action(self.action)
        self._act_t = now
        self._set(State.ACTING, f"performing {self.action} at {self.target.range:.2f} m")
