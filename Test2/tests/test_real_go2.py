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
