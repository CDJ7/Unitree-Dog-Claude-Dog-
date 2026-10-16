"""Drawing helpers: boxes with class/confidence/distance, optional LiDAR overlay, FPS."""
import cv2
import numpy as np


def _text(img, s, org, color=(0, 255, 0), scale=0.55):
    """Text with a dark halo. The halo is the same thickness-1 text shifted by 1 px in 8 directions: a thicker
    outline pass gets different letter spacing in OpenCV and smears into a doubled label."""
    x, y = org
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                cv2.putText(img, s, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _range_colors(r, lo=0.5, hi=8.0):
    t = np.clip((np.asarray(r) - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def draw_lidar_overlay(img, uvr):
    """uvr: (N,3) of pixel u, pixel v, range. Near = blue, far = red. Misalignment with edges = bad calibration."""
    if uvr is None or len(uvr) == 0:
        return
    colors = _range_colors(uvr[:, 2])
    for (u, v, _), c in zip(uvr, colors):
        cv2.circle(img, (int(u), int(v)), 2, tuple(int(x) for x in c), -1)


def draw_detections(frame, detections, infer_fps=0.0, display_fps=0.0, lidar_uvr=None,
                    show_lidar=False, status="", mission_text=""):
    img = frame.copy()
    if show_lidar:
        draw_lidar_overlay(img, lidar_uvr)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        gt = getattr(d, "source", "yolo") == "ground_truth"
        color = (255, 0, 255) if gt else ((0, 200, 255) if d.distance_m is None else (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        dist = "depth: n/a" if d.distance_m is None else f"{d.distance_m:.2f} m"
        label = f"GT {d.name} {dist}" if gt else f"{d.name} {d.conf:.2f} {dist}"
        h, w = img.shape[:2]
        ly = y1 - 8 if y1 > 70 else y1 + 18          # inside the box when it touches the top, clear of the status text
        _text(img, label, (min(x1, max(w - 12 * len(label), 4)), ly), color)
        if d.xyz_body is not None:
            x, y, z = d.xyz_body
            pos = f"fwd {x:+.2f} left {y:+.2f} up {z:+.2f}"
            _text(img, pos, (min(x1, max(w - 8 * len(pos), 4)), min(y2 + 16, h - 4)), color, 0.45)
    _text(img, f"infer {infer_fps:4.1f} fps  display {display_fps:4.1f} fps", (8, 20))
    if status:
        _text(img, status, (8, 40), (0, 200, 255))
    if any(getattr(d, "source", "yolo") == "ground_truth" for d in detections):
        _text(img, "GROUND TRUTH LABELS (from the simulator, not YOLO)", (8, 60), (255, 0, 255))
    if mission_text:
        _text(img, mission_text, (8, img.shape[0] - 34), (255, 255, 0), 0.6)
    _text(img, "L: LiDAR overlay   Esc: quit", (8, img.shape[0] - 10), (200, 200, 200), 0.45)
    return img
