#!/usr/bin/env python3
"""Live zone recognition on the REAL Go2, recorded to an MP4 for the video (real-world Q1, step 4).

    python real_go2/live_zone.py --source sdk --iface eth0
    python real_go2/live_zone.py --source webrtc --ip 192.168.8.181
    python real_go2/live_zone.py --source fake --headless --seconds 20          # test without a robot

Every new camera frame is preprocessed exactly as in training (centred 4:3 crop, 320 x 240, 168 x 224,
ImageNet normalisation) and classified by results/real_q1/resnet18/best.pt. The window shows the camera
and a panel with the predicted zone, every zone's confidence, a 1 s majority vote and, when you tell it
the truth, a running accuracy. Keys (click the window first):
  1..N  the true zone right now (0 = unknown)     V  start / stop the MP4     ESC quit
Driving uses Test 1's teleop keys (R stand, W/A/S/D/Q/E move, SPACE stop) unless --no-control.
Writes results/real_q1/live_<time>.mp4 (10 fps, real time) and live_<time>.csv (one row per classified frame).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))
from real_go2.sources import add_source_args, crop_resize, cycle_speed, open_link, start_teleop  # noqa: E402

PANEL_W, VIEW_W, VIEW_H = 330, 768, 432          # composite video 1098 x 432 (camera shown 16:9)
COLORS = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232), (0, 131, 0)]


class Classifier:
    def __init__(self, ckpt, crop):
        from q1_supervised.data import normalize
        from q1_supervised.evaluate import load_model
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.classes = list(ck["classes"])
        self.model, _ = load_model(ckpt, len(self.classes))
        self.h, self.w = ck["image_hw"]
        self.crop, self._norm = crop, normalize
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

    @torch.no_grad()
    def __call__(self, bgr):
        small = crop_resize(bgr, self.crop)                                  # as saved by record_real.py
        rgb = cv2.cvtColor(cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
        return self.model(self._norm(x)).softmax(1)[0].numpy()


def put_text(img, text, org, scale=0.55, color=(235, 235, 235), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def panel(classes, prob, smooth_k, true_k, acc, fps, infer_ms, recording):
    p = np.full((VIEW_H, PANEL_W, 3), 24, np.uint8)
    put_text(p, "Zone recognition - real Go2", (12, 26), 0.6, (255, 255, 255), 1)
    put_text(p, "ResNet18 (ImageNet), fine-tuned on this room", (12, 46), 0.42, (170, 170, 170))
    k = int(np.argmax(prob))
    put_text(p, "predicted:", (12, 82), 0.5, (170, 170, 170))
    put_text(p, classes[k], (12, 112), 0.95, COLORS[k % len(COLORS)], 2)
    y = 140
    for i, c in enumerate(classes):                                   # confidence bar per zone
        w = int(prob[i] * (PANEL_W - 130))
        cv2.rectangle(p, (100, y - 11), (100 + max(w, 1), y + 3), COLORS[i % len(COLORS)], -1)
        put_text(p, c[:11], (12, y), 0.45)
        put_text(p, f"{prob[i]:.2f}", (PANEL_W - 52, y), 0.45)
        y += 22
    y += 8
    put_text(p, f"1 s vote: {classes[smooth_k]}", (12, y), 0.5)
    y += 24
    if true_k is not None and true_k >= 0:
        ok = smooth_k == true_k
        put_text(p, f"true zone: {classes[true_k]}", (12, y), 0.5, (90, 230, 90) if ok else (90, 90, 240))
    else:
        put_text(p, "true zone: (press 1-%d)" % len(classes), (12, y), 0.45, (150, 150, 150))
    y += 24
    if acc[1]:
        put_text(p, f"live accuracy {acc[0] / acc[1]:.3f} ({acc[1]} labelled frames)", (12, y), 0.45)
    put_text(p, f"{fps:4.1f} fps   inference {infer_ms:4.0f} ms", (12, VIEW_H - 34), 0.45, (170, 170, 170))
    put_text(p, "REC" if recording else "not recording (V)", (12, VIEW_H - 12), 0.5,
             (60, 60, 235) if recording else (150, 150, 150), 1)
    if recording:
        cv2.circle(p, (PANEL_W - 22, VIEW_H - 17), 7, (60, 60, 235), -1)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--config", default=str(TEST2 / "real_go2/real_config.yaml"))
    ap.add_argument("--model", help="checkpoint (default: <results_dir>/resnet18/best.pt)")
    ap.add_argument("--headless", action="store_true", help="no window; records the MP4 from the start")
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
    tele = start_teleop(link, go2cfg, a, keyboard=not a.headless)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4_path, csv_path = out / f"live_{stamp}.mp4", out / f"live_{stamp}.csv"
    writer, recording = None, a.headless
    fh = open(csv_path, "w", newline="")
    log = csv.writer(fh)
    log.writerow(["t", "pred", "pred_zone", "conf", "vote", "true_zone"] + [f"p_{c}" for c in classes])
    votes = deque()                        # (time, pred) of the last second
    true_k, acc = -1, [0, 0]
    prob, smooth_k, infer_ms = np.ones(len(classes)) / len(classes), 0, 0.0
    t0 = time.monotonic()
    next_video, last_seq, n_frames = t0, -1, 0
    try:
        while True:
            img, seq = link.frame()
            now = time.monotonic()
            if a.source == "fake" and a.headless and link.hint.get()[0] is not None:
                h = link.hint.get()[0]
                true_k = h if h < len(classes) else -1
            if img is not None and seq != last_seq:
                last_seq = seq
                ti = time.perf_counter()
                prob = clf(img)
                infer_ms = 1000 * (time.perf_counter() - ti)
                k = int(np.argmax(prob))
                votes.append((now, k))
                while votes and now - votes[0][0] > 1.0:
                    votes.popleft()
                smooth_k = int(np.bincount([v for _, v in votes], minlength=len(classes)).argmax())
                if true_k >= 0:
                    acc[0] += int(k == true_k)
                    acc[1] += 1
                log.writerow([round(now - t0, 3), k, classes[k], round(float(prob[k]), 4), classes[smooth_k],
                              classes[true_k] if true_k >= 0 else ""] + [round(float(v), 4) for v in prob])
                n_frames += 1
            view = np.zeros((VIEW_H, VIEW_W, 3), np.uint8) if img is None else cv2.resize(img, (VIEW_W, VIEW_H))
            if img is None:
                put_text(view, "waiting for the camera ...", (250, 220), 0.8)
            if tele:
                put_text(view, f"speed {tele.speed_name.upper()} (G)", (12, VIEW_H - 14), 0.5)
            comp = np.hstack([view, panel(classes, prob, smooth_k, true_k, acc, link.fps(), infer_ms, recording)])
            if recording:
                if writer is None:
                    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), a.video_fps,
                                             (comp.shape[1], comp.shape[0]))
                while now >= next_video:                     # real-time video: one frame per 1/video_fps
                    writer.write(comp)
                    next_video += 1.0 / a.video_fps
            else:
                next_video = now
            if a.seconds and now - t0 >= a.seconds:
                break
            if a.headless:
                time.sleep(0.005)
                continue
            cv2.imshow("Go2 live zone recognition", comp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or (tele and tele.quit_event.is_set()):
                break
            if ord("1") <= key < ord("1") + len(classes):
                true_k = key - ord("1")
            elif key == ord("0"):
                true_k = -1
            elif key in (ord("v"), ord("V")):
                recording = not recording
            elif key in (ord("g"), ord("G")) and tele:
                cycle_speed(tele, go2cfg)
    except KeyboardInterrupt:
        pass
    finally:
        if tele:
            tele.shutdown()
        link.close()
        fh.close()
        if writer is not None:
            writer.release()
        if not a.headless:
            cv2.destroyAllWindows()
        summary = {"frames_classified": n_frames, "labelled_frames": acc[1],
                   "live_accuracy": round(acc[0] / acc[1], 4) if acc[1] else None,
                   "video": mp4_path.name if writer is not None else None, "log": csv_path.name,
                   "source": link.name, "classes": classes}
        (out / f"live_{stamp}.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
