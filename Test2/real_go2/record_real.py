#!/usr/bin/env python3
"""Record labelled front-camera frames from the REAL Go2 while you drive it (real-world Q1, step 1).

    python real_go2/record_real.py --source sdk --iface eth0                  # Ethernet (Go2 EDU)
    python real_go2/record_real.py --source webrtc --ip 192.168.8.181          # Wi-Fi
    python real_go2/record_real.py --source fake --headless --max-frames 300   # test without a robot

Driving (Test 1 teleop, keys work anywhere): R stand up, W/S forward/back, A/D sideways, Q/E turn,
SPACE stop, F lie down, ESC quit. Labelling (click the recorder window first):
  1..N  "the dog is now in zone k" (names from real_config.yaml `classes`), 0 = no zone (frames not saved)
  P     pause / resume saving
Each session becomes data_real/run_XXX/{images/, states.csv, metadata.json}, the same layout as the
simulated Q2 runs, so q1_supervised/train.py trains on it unchanged. Rows are written as they come in, so
a crash or disconnect keeps everything recorded so far.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))
from real_go2.sources import add_source_args, crop_resize, open_link, start_teleop  # noqa: E402

COLS = ["t", "frame", "zone_id", "zone", "cmd_vx", "cmd_vy", "cmd_yaw_rate", "base_x", "base_y", "base_yaw"]


def next_run_dir(root):
    root.mkdir(parents=True, exist_ok=True)
    k = 0
    while (root / f"run_{k:03d}").exists():
        k += 1
    return root / f"run_{k:03d}"


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--config", default=str(TEST2 / "real_go2/real_config.yaml"))
    ap.add_argument("--headless", action="store_true", help="no window (fake source: auto-labels from the sim zone)")
    ap.add_argument("--max-frames", type=int, help="stop after saving this many frames")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(a.config).read_text())
    classes = cfg["classes"]
    data_root = TEST2 / cfg["data_dir"]
    run_dir = next_run_dir(data_root)
    (run_dir / "images").mkdir(parents=True)
    link, go2cfg = open_link(a)
    tele = start_teleop(link, go2cfg, a, keyboard=not a.headless)

    meta = {"run": run_dir.name, "scene": "real", "route": {"name": "teleop", "kind": "manual"},
            "source": link.name, "classes": classes, "record_hz": cfg["record_hz"], "crop": cfg["crop"],
            "image_size": [320, 240], "started": datetime.now().isoformat(timespec="seconds"),
            "robot": "Unitree Go2 (real)"}
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=1))   # rewritten with counts at the end
    fh = open(run_dir / "states.csv", "w", newline="")
    wr = csv.writer(fh)
    wr.writerow(COLS)
    counts = np.zeros(len(classes), int)
    zone, recording = -1, True
    period = 1.0 / cfg["record_hz"]
    t0 = time.monotonic()
    next_t, last_seq, n = t0, -1, 0
    win = "Go2 zone recorder"
    print(f"Recording to {run_dir}  |  zones: " + "  ".join(f"{i + 1}={c}" for i, c in enumerate(classes)))
    try:
        while True:
            img, seq = link.frame()
            now = time.monotonic()
            if a.source == "fake" and a.headless:                 # emulate the key presses
                zone = link.hint.get()[0] if link.hint.get()[0] is not None else -1
                zone = zone if zone < len(classes) else -1
            if img is not None and now >= next_t and seq != last_seq:
                next_t = max(next_t + period, now)
                if recording and zone >= 0:
                    small = crop_resize(img, cfg["crop"])
                    cv2.imwrite(str(run_dir / "images" / f"{n:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    snap = tele.snapshot() if tele else {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}
                    x, y, yaw = link.pose()
                    wr.writerow([round(now - t0, 3), n, zone, classes[zone], round(snap["vx"], 3), round(snap["vy"], 3),
                                 round(snap["vyaw"], 3), x, y, yaw])
                    fh.flush()
                    counts[zone] += 1
                    n += 1
                last_seq = seq
                if a.max_frames and n >= a.max_frames:
                    break
            if a.headless:
                if a.source == "fake" and link.done.is_set():
                    break
                time.sleep(0.01)
                continue
            # ---- window
            view = np.zeros((480, 854, 3), np.uint8) if img is None else cv2.resize(img, (854, 480))
            if img is not None and cfg["crop"] == "center43":   # show the part that is saved
                x0 = (854 - 640) // 2
                cv2.rectangle(view, (x0, 0), (x0 + 639, 479), (0, 200, 255), 1)
            label = classes[zone] if zone >= 0 else "-- no zone (not saving) --"
            col = (60, 220, 60) if (recording and zone >= 0) else (60, 60, 230)
            put_text(view, f"ZONE: {label}", (12, 36), 0.9, col, 2)
            put_text(view, ("REC" if recording else "PAUSED") + f"   saved {n}   camera {link.fps():.1f} fps", (12, 66), 0.6, col)
            put_text(view, "  ".join(f"{i + 1}:{c} {counts[i]}" for i, c in enumerate(classes)), (12, 466), 0.5)
            if tele:
                s = tele.snapshot()
                put_text(view, f"cmd vx {s['vx']:+.2f} vy {s['vy']:+.2f} wz {s['vyaw']:+.2f}  {s['status']}", (12, 94), 0.5)
            for i, line in enumerate(link.status_lines()[:2]):
                put_text(view, line, (12, 118 + 22 * i), 0.5)
            if img is None:
                put_text(view, "waiting for the camera ...", (280, 250), 0.8)
            cv2.imshow(win, view)
            k = cv2.waitKey(15) & 0xFF
            if k == 27 or (tele and tele.quit_event.is_set()):
                break
            if ord("1") <= k < ord("1") + len(classes):
                zone = k - ord("1")
            elif k == ord("0"):
                zone = -1
            elif k in (ord("p"), ord("P")):
                recording = not recording
    except KeyboardInterrupt:
        pass
    finally:
        if tele:
            tele.shutdown()
        link.close()
        fh.close()
        meta.update({"frames": int(n), "frames_per_zone": dict(zip(classes, map(int, counts))),
                     "duration_s": round(time.monotonic() - t0, 1)})
        (run_dir / "metadata.json").write_text(json.dumps(meta, indent=1))
        if not a.headless:
            cv2.destroyAllWindows()
        print(f"saved {n} frames -> {run_dir}  " + ", ".join(f"{c}: {k}" for c, k in zip(classes, counts)))


if __name__ == "__main__":
    main()
