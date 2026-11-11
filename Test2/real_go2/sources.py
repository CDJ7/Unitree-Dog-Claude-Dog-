"""Connections to the REAL Unitree Go2 for the real-world Q1 zone classifier.

Every source gives the same three things:
  * the latest front-camera frame (BGR numpy image) and its sequence number,
  * a SportClient-compatible object (Move / StopMove / StandUp / ...) that Test 1's keyboard
    teleop (go2_stream/teleop.py) drives, with its watchdog and speed limits,
  * the dog's own odometry pose (x, y, yaw) where the connection provides it (else NaN).

Sources:
  sdk     Ethernet (Go2 EDU, 192.168.123.x), Unitree SDK2 over DDS: Test 1's tested code path
          (go2_stream/camera.py VideoClient, teleop.py SportClient, lidar.py StateSource).
  webrtc  Wi-Fi, through the same WebRTC protocol the Unitree phone app uses, via the open-source
          `unitree_webrtc_connect` package (pip install unitree_webrtc_connect). Video + sport commands.
          Only one WebRTC client can be connected: close the Unitree app first.
  fake    No robot: replays the simulated Q2 sample frames at 5 Hz, to test this pipeline anywhere.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

TEST2 = Path(__file__).resolve().parents[1]
REPO = TEST2.parent
GO2_STREAM = REPO / "go2_stream"
if str(GO2_STREAM) not in sys.path:
    sys.path.insert(0, str(GO2_STREAM))      # Test 1 modules are flat (import camera, teleop, lidar)

log = logging.getLogger("real_go2")

OUT_W, OUT_H = 320, 240                      # same frame size as the simulated Q2 dataset


def crop_resize(img, crop="center43"):
    """Real Go2 frames are 16:9; the Q1 network was designed for 4:3 (like the simulated camera).
    center43: cut a centred 4:3 window, then resize to 320 x 240. none: resize the full frame."""
    if crop == "center43":
        h, w = img.shape[:2]
        tw = int(round(h * 4 / 3))
        if tw < w:
            x0 = (w - tw) // 2
            img = img[:, x0:x0 + tw]
    return cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


class Latest:
    """Thread-safe newest-value buffer (same idea as go2_stream.camera.LatestFrame)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._v, self._seq, self._t = None, 0, 0.0

    def put(self, v):
        with self._lock:
            self._v, self._seq, self._t = v, self._seq + 1, time.monotonic()

    def get(self):
        with self._lock:
            return self._v, self._seq, self._t


# ------------------------------------------------------------------------------------------- SDK
class SdkLink:
    """Ethernet + Unitree SDK2 (DDS). Reuses Test 1's camera thread and state subscriber."""

    name = "sdk"

    def __init__(self, iface, go2_cfg):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        log.info("Initialising DDS on %s", iface)
        ChannelFactoryInitialize(0, iface)
        from camera import CameraThread
        from lidar import StateSource
        self.camera = CameraThread(go2_cfg["camera"], iface)
        self.camera.start()
        self.state = StateSource(go2_cfg["state"])
        self.sport_client = None                # None -> Test 1's Teleop creates the real SportClient

    def frame(self):
        img, seq, _ = self.camera.buffer.get()
        return img, seq

    def fps(self):
        return self.camera.fps

    def pose(self):
        s, _, _ = self.state.sport.get()
        if s is None:
            return (math.nan, math.nan, math.nan)
        return (float(s.position[0]), float(s.position[1]), float(s.imu_state.rpy[2]))

    def status_lines(self):
        self.state.check_stale()
        return self.state.lines()

    def close(self):
        self.camera.stop()


# ----------------------------------------------------------------------------------------- WebRTC
class WebRTCSportClient:
    """SportClient look-alike for Test 1's Teleop, sending the same sport API requests over WebRTC.
    Called from the teleop control thread; the sends are scheduled on the WebRTC asyncio loop."""

    def __init__(self, link):
        self._link = link
        from unitree_webrtc_connect.constants import DATA_CHANNEL_TYPE, RTC_TOPIC, SPORT_CMD
        self._topic, self._cmd, self._req = RTC_TOPIC["SPORT_MOD"], SPORT_CMD, DATA_CHANNEL_TYPE["REQUEST"]
        self._n = 0

    def _send(self, name, parameter=None):
        """Fire-and-forget request (same message as pub_sub.publish_request_new, without waiting for the
        reply, so the 20 Hz Move stream is not slowed down)."""
        conn = self._link.conn
        if conn is None or not self._link.connected.is_set():
            return 1
        self._n += 1
        payload = {"header": {"identity": {"id": int(time.time() * 1000) % 2147483648 + self._n % 1000,
                                           "api_id": self._cmd[name]}},
                   "parameter": json.dumps(parameter) if parameter is not None else ""}
        self._link.loop.call_soon_threadsafe(conn.datachannel.pub_sub.publish_without_callback,
                                             self._topic, payload, self._req)
        return 0

    def Move(self, vx, vy, vyaw): return self._send("Move", {"x": float(vx), "y": float(vy), "z": float(vyaw)})
    def StopMove(self): return self._send("StopMove")
    def StandUp(self): return self._send("StandUp")
    def BalanceStand(self): return self._send("BalanceStand")
    def StandDown(self): return self._send("StandDown")
    def Damp(self): return self._send("Damp")
    def Sit(self): return self._send("Sit")
    def RiseSit(self): return self._send("RiseSit")
    def Hello(self): return self._send("Hello")
    def Stretch(self): return self._send("Stretch")
    def Content(self): return self._send("Content")


class WebRTCLink:
    """Wi-Fi link through unitree_webrtc_connect. ip: the dog's address on your Wi-Fi (STA mode, shown in
    the Unitree app); ap=True when your laptop is connected to the dog's own hotspot instead.
    aes_key: required on Go2 firmware >= 1.1.15 (see real_go2/README.md, "AES key")."""

    name = "webrtc"

    def __init__(self, ip=None, ap=False, aes_key=None, normal_mode=True, timeout_s=30.0):
        try:
            from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
        except ImportError as e:
            raise SystemExit("The Wi-Fi source needs `pip install unitree_webrtc_connect` "
                             f"(see real_go2/README.md). Import failed: {e}")
        if not ap and not ip:
            raise SystemExit("--source webrtc needs --ip <dog's Wi-Fi IP> (or --ap when on the dog's hotspot)")
        kw = {"aes_128_key": aes_key} if aes_key else {}
        method = WebRTCConnectionMethod.LocalAP if ap else WebRTCConnectionMethod.LocalSTA
        self.conn = UnitreeWebRTCConnection(method, **({} if ap else {"ip": ip}), **kw)
        self.frames = Latest()
        self.connected = threading.Event()
        self.error = None
        self._times = []
        self._fps = 0.0
        self.loop = asyncio.new_event_loop()
        self._normal_mode = normal_mode
        self._thread = threading.Thread(target=self._run, name="webrtc", daemon=True)
        self._thread.start()
        if not self.connected.wait(timeout_s):
            raise SystemExit(f"WebRTC connection to the Go2 did not come up within {timeout_s:.0f} s: "
                             f"{self.error or 'no answer'}. See real_go2/README.md, troubleshooting.")
        self.sport_client = WebRTCSportClient(self)

    async def _on_track(self, track):
        while True:
            frame = await track.recv()
            self.frames.put(frame.to_ndarray(format="bgr24"))
            now = time.monotonic()
            self._times = [t for t in self._times if now - t < 2.0] + [now]
            self._fps = (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6)

    def _run(self):
        asyncio.set_event_loop(self.loop)

        async def setup():
            try:
                await self.conn.connect()
                if self._normal_mode:
                    await self._ensure_normal_mode()
                self.conn.video.switchVideoChannel(True)
                self.conn.video.add_track_callback(self._on_track)
                self.connected.set()
            except Exception as e:          # noqa: BLE001 - reported to the main thread
                self.error = f"{type(e).__name__}: {e}"
                log.error("WebRTC connection failed: %s", self.error)

        self.loop.run_until_complete(setup())
        self.loop.run_forever()

    async def _ensure_normal_mode(self):
        """The sport commands are served in the 'normal' motion mode (as in the library's example)."""
        from unitree_webrtc_connect.constants import RTC_TOPIC
        r = await self.conn.datachannel.pub_sub.publish_request_new(RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001})
        try:
            mode = json.loads(r["data"]["data"])["name"]
        except Exception:                   # noqa: BLE001
            mode = None
        if mode != "normal":
            log.info("Switching motion mode from %s to 'normal' (the dog may stand up)", mode)
            await self.conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1002, "parameter": {"name": "normal"}})
            await asyncio.sleep(5)

    def frame(self):
        img, seq, _ = self.frames.get()
        return img, seq

    def fps(self):
        return self._fps

    def pose(self):
        return (math.nan, math.nan, math.nan)

    def status_lines(self):
        return ["wifi (webrtc)"]

    def close(self):
        async def bye():
            try:
                await self.conn.disconnect()
            except Exception:               # noqa: BLE001
                pass
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(bye(), self.loop).result(timeout=5)
            self.loop.call_soon_threadsafe(self.loop.stop)


# ------------------------------------------------------------------------------------------- fake
class FakeLink:
    """Replays simulated frames (Test2/data/sample) at 5 Hz, with their sim zone as a hint, so the
    record / train / live pipeline can be tested without a robot."""

    name = "fake"

    def __init__(self, hz=5.0, loop=True):
        import pandas as pd
        self.items = []
        for run in sorted((TEST2 / "data/sample").glob("run_*")):
            st = pd.read_csv(run / "states.csv", usecols=["frame", "zone_id"])
            st = st[st.frame >= 0]
            self.items += [(run / "images" / f"{f:06d}.jpg", int(z)) for f, z in zip(st.frame, st.zone_id)]
        self.frames = Latest()
        self.hint = Latest()
        self._stop = threading.Event()
        self._hz, self._loop = hz, loop
        self.done = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        from teleop import FakeSportClient
        self.sport_client = FakeSportClient()

    def _run(self):
        while not self._stop.is_set():
            for p, z in self.items:
                if self._stop.is_set():
                    return
                img = cv2.imread(str(p))
                img = cv2.resize(img, (1280, 720))       # look like a real 16:9 Go2 frame
                self.hint.put(z)
                self.frames.put(img)
                time.sleep(1.0 / self._hz)
            if not self._loop:
                self.done.set()
                return

    def frame(self):
        img, seq, _ = self.frames.get()
        return img, seq

    def fps(self):
        return self._hz

    def pose(self):
        return (math.nan, math.nan, math.nan)

    def status_lines(self):
        return ["fake source (simulated frames)"]

    def close(self):
        self._stop.set()


def add_source_args(ap):
    ap.add_argument("--source", choices=["sdk", "webrtc", "fake"], default="sdk",
                    help="sdk = Ethernet (Test 1 path, default); webrtc = Wi-Fi; fake = no robot")
    ap.add_argument("--iface", help="sdk: Ethernet interface on 192.168.123.x (default: go2_stream/config.yaml)")
    ap.add_argument("--ip", help="webrtc: the dog's IP on your Wi-Fi (STA mode)")
    ap.add_argument("--ap", action="store_true", help="webrtc: laptop is on the dog's own Wi-Fi hotspot")
    ap.add_argument("--aes-key", help="webrtc: per-device key, needed on Go2 firmware >= 1.1.15 "
                                      "(or set GO2_AES_KEY)")
    ap.add_argument("--no-control", action="store_true", help="camera only, do not drive the dog")
    ap.add_argument("--max-speed", type=float, default=0.3, help="teleop forward speed limit, m/s")
    ap.add_argument("--fake-hz", type=float, default=5.0, help="fake: replay rate (frames per second)")


def go2_config():
    import yaml
    return yaml.safe_load((GO2_STREAM / "config.yaml").read_text())


def open_link(a):
    """Open the chosen source. Returns (link, go2_stream config)."""
    import os
    cfg = go2_config()
    if a.source == "sdk":
        iface = a.iface or cfg["network"]["iface"]
        if not Path(f"/sys/class/net/{iface}").exists():
            raise SystemExit(f"No network interface '{iface}'. Available: {', '.join(os.listdir('/sys/class/net'))}. "
                             "Pass --iface (on WSL see real_go2/README.md, 'mirrored networking').")
        return SdkLink(iface, cfg), cfg
    if a.source == "webrtc":
        return WebRTCLink(ip=a.ip, ap=a.ap, aes_key=a.aes_key or os.environ.get("GO2_AES_KEY")), cfg
    return FakeLink(hz=a.fake_hz), cfg


def start_teleop(link, cfg, a, keyboard=True):
    """Test 1's keyboard teleop (W/S/A/D/Q/E, R stand, F down, SPACE stop, ESC quit) on this link."""
    if a.no_control:
        return None
    from teleop import Teleop
    tele = Teleop(cfg["control"], client=link.sport_client, max_speed=a.max_speed)
    tele.start(use_keyboard=keyboard)
    return tele
