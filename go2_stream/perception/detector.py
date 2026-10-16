"""Ultralytics YOLO wrapper."""
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("detector")


@dataclass
class Detection:
    cls_id: int
    name: str
    conf: float
    bbox: Tuple[float, float, float, float]          # x1, y1, x2, y2 in pixels
    distance_m: Optional[float] = None               # None -> "depth: n/a"
    xyz_cam: Optional[Tuple[float, float, float]] = None    # optical frame: x right, y down, z forward (m)
    xyz_body: Optional[Tuple[float, float, float]] = None   # body frame: x forward, y left, z up (m)
    n_depth_points: int = 0
    xyz_world: Optional[Tuple[float, float, float]] = None  # world frame (sim, or robot odom on the real dog), m
    source: str = "yolo"                                     # "yolo" or "ground_truth" (sim only, never mixed up)


def pick_device(requested="auto"):
    """'auto' -> first CUDA GPU if torch sees one, else CPU. Prints the choice."""
    if requested != "auto":
        log.info("Using device: %s (requested)", requested)
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda:0"
            log.info("Using device: %s (%s)", dev, torch.cuda.get_device_name(0))
            return dev
    except Exception as e:
        log.warning("torch CUDA check failed (%s)", e)
    log.info("Using device: cpu (no CUDA GPU available)")
    return "cpu"


class YoloDetector:
    def __init__(self, model="yolov8n.pt", conf=0.35, device="auto", classes=None, imgsz=640):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("ultralytics is not installed: pip install ultralytics") from e
        self.conf, self.imgsz = conf, imgsz
        self.device = pick_device(device)
        self.model = YOLO(model)
        self.names = self.model.names                     # {id: name}
        self.class_ids = self._resolve_classes(classes)
        log.info("Loaded %s (%d classes)%s", model, len(self.names),
                 f", filter: {classes}" if classes else "")

    def _resolve_classes(self, classes):
        if not classes:
            return None
        lookup = {n.lower(): i for i, n in self.names.items()}
        missing = [c for c in classes if str(c).lower() not in lookup]
        if missing:
            raise ValueError(f"Unknown class names {missing}. Valid names include: "
                             f"{', '.join(sorted(lookup)[:20])}, ...")
        return [lookup[str(c).lower()] for c in classes]

    def detect(self, frame_bgr):
        res = self.model.predict(frame_bgr, conf=self.conf, device=self.device, classes=self.class_ids,
                                 imgsz=self.imgsz, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for xyxy, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                              res.boxes.cls.cpu().numpy()):
            out.append(Detection(int(k), self.names[int(k)], float(c), tuple(float(v) for v in xyxy)))
        return out
