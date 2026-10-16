import copy
import math
import os
import sys
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from behavior.control import (Target, approach_command, candidates, corridor_obstacle_distance, pick_target,  # noqa: E402
                              validate_config)
from behavior.mission import Mission, State  # noqa: E402
from perception.detector import Detection  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config", "behaviors.yaml")))
P = CFG["safety"]


def fast_cfg(**safety):
    c = copy.deepcopy(CFG)
    c["safety"].update(settle_s=0.3, confirm_frames=3, **safety)
    c["actions"]["sit"]["hold_s"] = 0.3
    return c


# ----------------------------------------------------------------------------- pure functions
def tgt(x, y, cls="chair"):
    return Target(cls, x, y, 0.9, 0.0)


def test_approach_walks_forward_slowly_and_never_exceeds_the_cap():
    vx, vyaw, arrived = approach_command(tgt(3.0, 0.0), 0.9, P)
    assert 0 < vx <= P["max_vx"] and vyaw == 0 and not arrived
    vx, _, _ = approach_command(tgt(1.2, 0.0), 0.9, P)
    assert 0 < vx < P["max_vx"]                       # slows near the goal


def test_approach_turns_in_place_first_and_never_reverses_or_strafes():
    vx, vyaw, _ = approach_command(tgt(0.1, 2.0), 0.9, P)           # target 90 degrees to the left
    assert vx == 0 and 0 < vyaw <= P["max_vyaw"]
    vx, vyaw, _ = approach_command(tgt(0.1, -2.0), 0.9, P)          # to the right
    assert vx == 0 and -P["max_vyaw"] <= vyaw < 0
    vx, vyaw, _ = approach_command(tgt(-2.0, 0.1), 0.9, P)          # behind: turn, don't back up
    assert vx == 0 and abs(vyaw) > 0


def test_arrived_within_tolerance_and_never_closer_than_commanded():
    assert approach_command(tgt(0.95, 0.0), 0.9, P)[2]
    vx, _, arrived = approach_command(tgt(0.7, 0.0), 0.9, P)         # already past the stop distance
    assert arrived and vx == 0


def test_pick_target_nearest_then_sticks_to_the_same_one():
    a, b = tgt(2.0, 0.5), tgt(3.0, -1.0)
    assert pick_target([a, b]) is a
    prev = tgt(3.0, -0.9)
    assert pick_target([a, b], prev) is b                            # keeps tracking b, does not jump to the nearer a
    assert pick_target([a], prev) is None                            # b vanished: a is outside the gate, so no target


def test_candidates_filter_class_confidence_and_missing_positions():
    d = [Detection(0, "chair", 0.9, (0, 0, 1, 1), xyz_body=(2.0, 0.1, 0.4)),
         Detection(0, "chair", 0.3, (0, 0, 1, 1), xyz_body=(2.0, 0.1, 0.4)),
         Detection(0, "cup", 0.9, (0, 0, 1, 1), xyz_body=(1.0, 0.0, 0.4)),
         Detection(0, "chair", 0.9, (0, 0, 1, 1))]
    assert [c.cls for c in candidates(d, ["chair"], 0.5, 0.0)] == ["chair"]


def test_corridor_sees_obstacles_ahead_but_not_floor_target_or_sides():
    c = P["corridor"]
    wall = np.array([[0.8, 0.0, 0.2], [0.82, 0.1, 0.1], [0.85, -0.1, 0.3]])
    assert corridor_obstacle_distance(wall, c) == pytest.approx(0.85, abs=0.01)
    floor = np.array([[0.5, 0.0, -0.27]] * 10)
    assert corridor_obstacle_distance(floor, c) is None
    side = np.array([[0.8, 0.9, 0.2]] * 10)
    assert corridor_obstacle_distance(side, c) is None
    assert corridor_obstacle_distance(wall[:1], c) is None            # one stray return is ignored
    assert corridor_obstacle_distance(wall, c, exclude_xy=(0.82, 0.0)) is None   # that IS the target
    assert corridor_obstacle_distance(np.empty((0, 3)), c) is None


def test_config_validation_rejects_unsafe_settings_and_applies_the_stop_floor():
    bad = copy.deepcopy(CFG)
    bad["classes"]["chair"]["action"] = "backflip"
    with pytest.raises(ValueError):
        validate_config(bad)
    fast = copy.deepcopy(CFG)
    fast["safety"]["max_vx"] = 0.9
    with pytest.raises(ValueError):
        validate_config(fast)
    close = copy.deepcopy(CFG)
    close["classes"]["chair"]["stop_distance_m"] = 0.2
    assert validate_config(close)["chair"][1] == P["min_stop_distance_m"]


# ----------------------------------------------------------------------------- mission with a fake robot
class FakeTeleop:
    def __init__(self):
        self.standing = True
        self.quit_event = threading.Event()
        self.cmd, self.cmd_t = (0.0, 0.0, 0.0), 0.0
        self.actions, self.busy_until = [], 0.0
        self.set_calls = 0
        self.max_vx_seen = 0.0

    def set_autonomy(self, vx, vy, wz):
        self.cmd, self.cmd_t, self.set_calls = (vx, vy, wz), time.monotonic(), self.set_calls + 1
        self.max_vx_seen = max(self.max_vx_seen, abs(vx))
        assert vy == 0.0                                              # never strafes

    def clear_autonomy(self):
        self.cmd = (0.0, 0.0, 0.0)

    def request_action(self, name):
        self.actions.append(name)
        self.busy_until = time.monotonic() + 0.3

    def idle(self):
        return time.monotonic() > self.busy_until


class FakeWorld(threading.Thread):
    """Integrates the commands like Teleop would (with the 0.3 s deadman) and publishes detection results at 10 Hz."""

    def __init__(self, teleop, objects, start=(0.0, 0.0, 0.0)):
        super().__init__(daemon=True)
        self.t, self.objects = teleop, dict(objects)
        self.x, self.y, self.yaw = start
        self.latest, self.seq, self.publish, self.stop_flag = None, 0, True, False
        self.backwards = False

    def run(self):
        n = 0
        while not self.stop_flag:
            vx, _, wz = self.t.cmd if time.monotonic() - self.t.cmd_t <= 0.3 else (0.0, 0.0, 0.0)
            if vx < 0:
                self.backwards = True
            dt = 0.05
            self.yaw += wz * dt
            self.x += vx * math.cos(self.yaw) * dt
            self.y += vx * math.sin(self.yaw) * dt
            n += 1
            if self.publish and n % 2 == 0:
                dets = []
                for name, (ox, oy) in self.objects.items():
                    dx, dy = ox - self.x, oy - self.y
                    xb = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
                    yb = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy
                    if xb > 0.1 and abs(math.atan2(yb, xb)) < 0.9:             # inside the camera's view
                        dets.append(Detection(0, name.split("#")[0], 0.9, (0, 0, 1, 1), 1.0, xyz_body=(xb, yb, 0.4)))
                self.seq += 1
                self.latest = NS(detections=dets, wall_time=time.time(), seq=self.seq)
            time.sleep(0.05)

    def dist_to(self, name):
        ox, oy = self.objects[name]
        return math.hypot(ox - self.x, oy - self.y)


def make(cfg=None, objects=None, start=(0.0, 0.0, 0.0), **mission_kw):
    t = FakeTeleop()
    w = FakeWorld(t, objects if objects is not None else {"chair#1": (3.0, 0.5)}, start)
    m = Mission(cfg or fast_cfg(), lambda: w.latest, t, **mission_kw)
    w.start()
    m.start()
    return t, w, m


def wait_for(cond, timeout=20.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


def ready(m, timeout=5.0):
    return wait_for(lambda: m.candidate is not None and m._seen >= m.p["confirm_frames"], timeout)


def finish(t, w, m):
    m.stop(); w.stop_flag = True


def test_full_mission_approaches_stops_at_the_distance_sits_then_rises():
    t, w, m = make()
    assert ready(m)
    ok, msg = m.go()
    assert ok, msg
    assert wait_for(lambda: m.state == State.DONE), m.status_text()
    finish(t, w, m)
    stop = m.table["chair"][1]
    assert abs(w.dist_to("chair#1") - stop) < 0.25                     # stopped near the configured distance
    assert w.dist_to("chair#1") > stop - 0.2                           # and did not run into it
    assert t.actions == ["sit", "rise"]
    assert t.max_vx_seen <= P["max_vx"] + 1e-9 and not w.backwards


def test_nothing_moves_without_the_go_ahead():
    t, w, m = make()
    assert ready(m)
    time.sleep(1.0)
    finish(t, w, m)
    assert t.set_calls == 0 and t.actions == [] and w.x == 0.0


def test_go_is_refused_unless_it_is_safe():
    t, w, m = make(objects={"chair#1": (6.0, 0.0)})                    # too far
    assert ready(m)
    ok, msg = m.go()
    assert not ok and "away" in msg
    finish(t, w, m)

    t, w, m = make()
    assert ready(m)
    t.standing = False
    ok, msg = m.go()
    assert not ok and "standing" in msg
    finish(t, w, m)

    t, w, m = make(objects={"chair#1": (0.6, 0.0)})                    # already closer than the stop distance
    assert ready(m)
    ok, msg = m.go()
    assert not ok and "closer" in msg
    finish(t, w, m)

    t, w, m = make(allow_autonomy=False)
    assert ready(m)
    ok, msg = m.go()
    assert not ok and "disabled" in msg
    finish(t, w, m)

    t, w, m = make(objects={})                                          # nothing visible
    time.sleep(0.5)
    ok, msg = m.go()
    assert not ok
    finish(t, w, m)


def test_a_key_press_aborts_and_the_robot_stops():
    t, w, m = make()
    assert ready(m)
    assert m.go()[0]
    assert wait_for(lambda: w.x > 0.3)
    m.manual_input("s")
    assert m.state == State.ABORTED and t.cmd == (0.0, 0.0, 0.0)
    x = w.x
    time.sleep(0.6)
    finish(t, w, m)
    assert abs(w.x - x) < 0.08 and t.actions == []


def test_target_lost_aborts_without_acting():
    t, w, m = make()
    assert ready(m)
    assert m.go()[0]
    assert wait_for(lambda: w.x > 0.3)
    w.objects.clear()                                                  # chair disappears from view
    assert wait_for(lambda: m.state == State.ABORTED, 5.0), m.status_text()
    finish(t, w, m)
    assert "lost" in m.message and t.actions == []


def test_stale_detections_stop_the_dog_immediately():
    t, w, m = make()
    assert ready(m)
    assert m.go()[0]
    assert wait_for(lambda: w.x > 0.3)
    w.publish = False                                                   # detector stalls
    assert wait_for(lambda: m.last_cmd == (0.0, 0.0, 0.0), 2.0)
    x = w.x
    time.sleep(0.5)
    assert abs(w.x - x) < 0.1
    assert wait_for(lambda: m.state == State.ABORTED, 4.0)
    finish(t, w, m)


def test_blocked_path_stops_then_aborts():
    t, w, m = make(cfg=fast_cfg(blocked_abort_s=0.6), clearance_fn=lambda exclude: 0.3)
    assert ready(m)
    assert m.go()[0]
    assert wait_for(lambda: m.state == State.ABORTED, 5.0), m.status_text()
    finish(t, w, m)
    assert "blocked" in m.message and w.x < 0.05 and t.actions == []


def test_clear_path_does_not_interfere():
    t, w, m = make(clearance_fn=lambda exclude: None)
    assert ready(m)
    assert m.go()[0]
    assert wait_for(lambda: m.state == State.DONE)
    finish(t, w, m)


def test_dry_run_sends_nothing_to_the_robot():
    c = fast_cfg()
    c["dry_run_report_s"] = 1.5
    t, w, m = make(cfg=c, dry_run=True)
    assert ready(m)
    assert m.go()[0]
    assert wait_for(lambda: m.state == State.DONE, 6.0), m.status_text()
    computed = m.last_cmd                                               # read before stop() zeroes it
    finish(t, w, m)
    assert t.set_calls == 0 and t.actions == [] and w.x == 0.0
    assert computed[0] > 0                                              # it did compute a forward command


def test_confirm_before_action_waits_for_the_key():
    t, w, m = make(cfg=fast_cfg(confirm_before_action=True))
    assert ready(m)
    assert m.go()[0]
    assert wait_for(lambda: m.state == State.AWAIT_CONFIRM, 20.0), m.status_text()
    time.sleep(0.8)
    assert t.actions == []
    m.confirm()
    assert wait_for(lambda: m.state == State.DONE)
    finish(t, w, m)
    assert t.actions[0] == "sit"


def test_dog_lying_down_mid_mission_aborts():
    t, w, m = make()
    assert ready(m)
    assert m.go()[0]
    assert wait_for(lambda: w.x > 0.2)
    t.standing = False
    assert wait_for(lambda: m.state == State.ABORTED, 3.0)
    finish(t, w, m)


def test_two_chairs_picks_the_nearest_and_unknown_class_is_rejected():
    t, w, m = make(objects={"chair#1": (4.0, 0.0), "chair#2": (2.5, 1.0)})
    assert ready(m)
    assert m.candidate.range == pytest.approx(math.hypot(2.5, 1.0), abs=0.1)
    finish(t, w, m)
    with pytest.raises(ValueError):
        Mission(CFG, lambda: None, FakeTeleop(), classes=["toaster"])


def test_go_again_after_done_rearms_instead_of_driving_off():
    t, w, m = make()
    assert ready(m)
    assert m.go()[0]
    assert wait_for(lambda: m.state == State.DONE)
    ok, msg = m.go()
    assert not ok and m.state == State.READY
    finish(t, w, m)
