"""Camera zone classifier: one head-camera frame -> P(zone), from a Test 2 Q1 checkpoint.

Checkpoints (ResNet18 fine-tuned from ImageNet, see Test2/REPORT.md sections 5 and 7):
  ../Test2/results/q1/resnet18/best.pt        simulation, house 185: victim_area, bed_area, lounge_area,
                                              desk_entry_area, bathroom (in git)
  ../Test2/results/real_q1/resnet18/best.pt   the real lab: lab_1, lab_2, open_area, shelves
                                              (trained on the lab laptop; not in git)
Preprocessing is the one used in training: a 16:9 real Go2 frame is first cropped to its centred 4:3
window and resized to 320 x 240 (as Test2 record_real.py saved it); then resize to the checkpoint's
image_hw, RGB, ImageNet normalisation.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PHASE2 = Path(__file__).resolve().parents[1]
SIM_MODEL = PHASE2.parent / "Test2/results/q1/resnet18/best.pt"
REAL_MODEL = PHASE2.parent / "Test2/results/real_q1/resnet18/best.pt"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def center43(img):
    """Centred 4:3 crop of a wider frame, resized to 320 x 240 (a 4:3 frame is only resized)."""
    import cv2
    h, w = img.shape[:2]
    if w * 3 > h * 4 + 4:
        cw = int(round(h * 4 / 3))
        x0 = (w - cw) // 2
        img = img[:, x0:x0 + cw]
    return cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)


class ZoneClassifier:
    def __init__(self, ckpt, threads=None):
        import torch
        import torchvision
        ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if ck.get("model") != "resnet18":
            raise ValueError(f"{ckpt}: expected a resnet18 checkpoint, got {ck.get('model')}")
        self.classes = list(ck["classes"])
        self.h, self.w = ck["image_hw"]
        m = torchvision.models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, len(self.classes))
        m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
        self.model = m.eval()
        self.torch = torch
        torch.set_num_threads(threads or max(1, (os.cpu_count() or 2) - 1))

    def __call__(self, rgb):
        """P(zone) for one RGB uint8 frame (any size; 16:9 frames are centre-cropped to 4:3)."""
        import cv2
        x = cv2.resize(center43(rgb), (self.w, self.h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        t = self.torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        with self.torch.no_grad():
            return self.model(t).softmax(1)[0].numpy()
