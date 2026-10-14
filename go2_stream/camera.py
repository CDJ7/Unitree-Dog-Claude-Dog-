"""Front-camera capture: SDK VideoClient (JPEG over RPC) or multicast H.264 via GStreamer/OpenCV."""
import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("camera")


class LatestFrame:
    """Thread-safe latest-value buffer: writers never block, readers get the newest frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._stamp = 0.0

    def put(self, frame):
        with self._lock:
            self._frame, self._seq, self._stamp = frame, self._seq + 1, time.monotonic()

    def get(self):
        with self._lock:
            return self._frame, self._seq, self._stamp


class CameraThread(threading.Thread):
    def __init__(self, cfg, iface):
        super().__init__(name="camera", daemon=True)
        self.cfg, self.iface = cfg, iface
        self.buffer = LatestFrame()
        self.fps = 0.0
        self._stop_evt = threading.Event()
        self._warned = False

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            if self.cfg["mode"] == "multicast":
                self._run_multicast()
            else:
                self._run_videoclient()
        except Exception:
            log.exception("camera thread crashed")

    def _tick(self, times):
        now = time.monotonic()
        times.append(now)
        while times and now - times[0] > 2.0:
            times.pop(0)
        self.fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0

    def _run_videoclient(self):
        from unitree_sdk2py.go2.video.video_client import VideoClient
        client = VideoClient()
        client.SetTimeout(3.0)
        client.Init()
        log.info("VideoClient started (service 'videohub', JPEG frames via RPC)")
        times, last_ok, fails = [], time.monotonic(), 0
        while not self._stop_evt.is_set():
            code, data = client.GetImageSample()
            if code != 0:
                fails += 1
                if time.monotonic() - last_ok > self.cfg["no_data_warn_s"] and fails % 5 == 1:
                    log.warning("No camera frames for >%.0fs (VideoClient code %s). Check: robot is EDU and "
                                "powered, `ping 192.168.123.161`, correct --iface, and that no other client "
                                "(e.g. the app's video) is hogging the service.",
                                self.cfg["no_data_warn_s"], code)
                time.sleep(0.2)
                continue
            img = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("Received a frame that OpenCV could not decode")
                continue
            fails, last_ok = 0, time.monotonic()
            self.buffer.put(img)
            self._tick(times)

    def _run_multicast(self):
        mc = self.cfg["multicast"]
        iface = mc.get("iface") or self.iface
        pipe = " ".join(mc["pipeline"].split()).format(address=mc["address"], port=mc["port"], iface=iface)
        log.info("GStreamer pipeline: %s", pipe)
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            log.error("Could not open the GStreamer pipeline. OpenCV must be built with GStreamer "
                      "(`python -c \"import cv2;print(cv2.getBuildInformation())\" | grep -i gstreamer`; the pip "
                      "opencv-python wheel does NOT include it -> use system python3-opencv + gstreamer1.0-libav). "
                      "Also verify the group/port with tcpdump (see README).")
            return
        times, last_ok = [], time.monotonic()
        while not self._stop_evt.is_set():
            ok, img = cap.read()
            if not ok:
                if time.monotonic() - last_ok > self.cfg["no_data_warn_s"]:
                    log.warning("No multicast video for >%.0fs on %s:%s via %s. Check the multicast route/"
                                "interface (`ip route`, tcpdump) and the address/port in config.yaml.",
                                self.cfg["no_data_warn_s"], mc["address"], mc["port"], iface)
                    last_ok = time.monotonic()
                time.sleep(0.05)
                continue
            last_ok = time.monotonic()
            self.buffer.put(img)
            self._tick(times)
        cap.release()


def draw_overlay(img, fps, cmd, scale, status, extra_lines=()):
    """Draw FPS, current command velocity and status text onto a copy of the frame."""
    out = img.copy()
    lines = [f"FPS {fps:4.1f}",
             f"cmd vx {cmd[0]:+.2f} vy {cmd[1]:+.2f} wz {cmd[2]:+.2f}  scale {scale:.1f}",
             f"status: {status}", *extra_lines]
    y = 22
    for text in lines:
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        y += 20
    return out
