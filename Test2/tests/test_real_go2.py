"""Real-Go2 Q1 tools: preprocessing, the by-recording-order split, and the Wi-Fi sport command format."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_go2.sources import crop_resize  # noqa: E402
from q1_supervised.data import split_runs  # noqa: E402


def test_crop_resize_16x9_to_4x3():
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, :160] = 255                                   # left strip lies outside the centred 4:3 window
    out = crop_resize(img, "center43")
    assert out.shape == (240, 320, 3) and out.max() == 0
    assert crop_resize(img, "none").shape == (240, 320, 3)


def test_split_by_recording_order():
    frames = pd.DataFrame({"run": [f"run_00{i}" for i in (0, 0, 1, 2, 3, 3)]})
    s = split_runs(frames, {"split": {"mode": "by_order"}})
    assert s == {"train": ["run_000", "run_001"], "val": ["run_002"], "test": ["run_003"]}


def test_webrtc_move_message(monkeypatch):
    """The Wi-Fi driver sends the same request the library's own publish_request_new would build."""
    consts = types.ModuleType("unitree_webrtc_connect.constants")
    consts.RTC_TOPIC = {"SPORT_MOD": "rt/api/sport/request"}
    consts.SPORT_CMD = {k: v for k, v in [("Move", 1008), ("StopMove", 1003), ("StandUp", 1004),
                                          ("BalanceStand", 1002), ("StandDown", 1005), ("Damp", 1001),
                                          ("Sit", 1009), ("RiseSit", 1010), ("Hello", 1016),
                                          ("Stretch", 1017), ("Content", 1020)]}
    consts.DATA_CHANNEL_TYPE = {"REQUEST": "req"}
    monkeypatch.setitem(sys.modules, "unitree_webrtc_connect", types.ModuleType("unitree_webrtc_connect"))
    monkeypatch.setitem(sys.modules, "unitree_webrtc_connect.constants", consts)
    from real_go2.sources import WebRTCSportClient
    sent = []
    pub = types.SimpleNamespace(publish_without_callback=lambda topic, data, kind: sent.append((topic, data, kind)))
    link = types.SimpleNamespace(conn=types.SimpleNamespace(datachannel=types.SimpleNamespace(pub_sub=pub)),
                                 connected=types.SimpleNamespace(is_set=lambda: True),
                                 loop=types.SimpleNamespace(call_soon_threadsafe=lambda f, *a: f(*a)))
    c = WebRTCSportClient(link)
    assert c.Move(0.3, -0.1, 0.5) == 0 and c.StopMove() == 0
    topic, data, kind = sent[0]
    assert topic == "rt/api/sport/request" and kind == "req"
    assert data["header"]["identity"]["api_id"] == 1008
    assert json.loads(data["parameter"]) == {"x": 0.3, "y": -0.1, "z": 0.5}
    assert sent[1][1]["header"]["identity"]["api_id"] == 1003 and sent[1][1]["parameter"] == ""


def test_summarize_counts_robot_missions_only(tmp_path):
    """summarize_real.goto_table: counts arrived/failed requests from robot runs, skips fake and cancelled."""
    import json
    from real_go2 import summarize_real
    runs = {"goto_1.json": {"source": "sdk", "lidar": True, "missions": [
                {"target": "shelves", "result": "arrived", "reason": "arrived", "legs": 3},
                {"target": "shelves", "result": "failed", "reason": "shelves not seen in any direction (best 0.1)",
                 "legs": 10},
                {"target": "lab_1", "result": "cancelled", "reason": "cancelled by key ' '", "legs": 1}]},
            "goto_2.json": {"source": "fake", "missions": [{"target": "lab_1", "result": "arrived", "legs": 1}]}}
    for name, d in runs.items():
        (tmp_path / name).write_text(json.dumps(d))
    table = summarize_real.goto_table(tmp_path)
    assert "| shelves | 2 | 1 | 3 | shelves not seen in any direction x1 |" in table
    assert "**1/2** arrived" in table and "lab_1" not in table


def test_speed_presets_cycle_and_respect_hard_caps():
    """--speed / G: slow -> medium -> fast -> slow; each one changes the teleop limits; Test 1's hard caps still win."""
    import yaml
    from real_go2.sources import GO2_STREAM, SPEED_PRESETS, cycle_speed, set_speed
    cfg = yaml.safe_load((GO2_STREAM / "config.yaml").read_text())
    c = cfg["control"]

    class Tele:                                     # stands in for Test 1's Teleop: only .limits is used
        limits = (0, 0, 0)

    t = Tele()
    set_speed(t, c, "slow")
    assert t.limits == (0.25, 0.75 * 0.25, 1.2)              # turning is fast in every preset
    seen = []
    for _ in range(3):
        seen.append(cycle_speed(t, cfg))
        assert t.limits[0] <= c["hard_cap_vx"] and t.limits[1] <= c["hard_cap_vy"] and t.limits[2] <= c["hard_cap_vyaw"]
    assert seen == ["medium", "fast", "slow"]
    set_speed(t, c, "fast")
    assert t.limits[0] > SPEED_PRESETS["slow"]["vx"]
    assert all(p["vyaw"] == 1.2 for p in SPEED_PRESETS.values())
    set_speed(t, c, "fast", max_speed=0.3)          # --max-speed overrides the forward speed only
    assert t.limits[0] == 0.3 and t.limits[2] == SPEED_PRESETS["fast"]["vyaw"]


def test_goto_zone_speed_presets_keep_no_lidar_cap():
    import yaml
    from real_go2.goto_zone import ZoneNavigator, apply_nav_speed
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "real_go2/real_config.yaml").read_text())
    nav = ZoneNavigator(cfg["classes"], cfg["navigation"], have_lidar=True)
    apply_nav_speed(nav, cfg["navigation"], "fast")
    assert nav.nav["approach_speed"] == 0.45 and nav.nav["scan_rate"] == 0.7
    blind = ZoneNavigator(cfg["classes"], cfg["navigation"], have_lidar=False)
    apply_nav_speed(blind, cfg["navigation"], "fast")
    assert blind.nav["approach_speed"] <= 0.15
