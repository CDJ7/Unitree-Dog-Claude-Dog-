"""GROUND TRUTH labels from the simulator. This is NOT detection: it reads object identities straight from
the scene. It exists so mapping and accuracy tests can run even when YOLO does not recognise the sim's
rendered shapes. Detections it produces carry source="ground_truth" and are drawn/labelled as such."""
import numpy as np

from perception.depth_fusion import world_to_optical
from perception.detector import Detection


class GroundTruthDetector:
    """Same .detect(frame) interface as YoloDetector; needs the SimWorld to look up the frame's pose and depth."""

    def __init__(self, world, min_px=6, min_visible_frac=0.10):
        self.world, self.min_px, self.min_visible_frac = world, min_px, min_visible_frac
        self.device = "sim"

    def detect(self, frame):
        b = self.world.bundle_for(frame)
        if b is None:
            return []
        cm = self.world.camera_model
        h, w = b.depth.shape
        out = []
        for o in self.world.objects:
            p = world_to_optical(o.corners, b.cam_pos, b.cam_mat)
            front = p[:, 2] > 0.05
            if not front.any():
                continue
            if not front.all():                    # object straddles the image plane: clip to the points in front
                p = p[front]
            u = cm.fx * p[:, 0] / p[:, 2] + cm.cx
            v = cm.fy * p[:, 1] / p[:, 2] + cm.cy
            x1, x2 = max(u.min(), 0.0), min(u.max(), w - 1.0)
            y1, y2 = max(v.min(), 0.0), min(v.max(), h - 1.0)
            if x2 - x1 < self.min_px or y2 - y1 < self.min_px:
                continue
            # occlusion: is something at the object's depth visible inside the box?
            zlo, zhi = p[:, 2].min() - 0.15, p[:, 2].max() + 0.15
            patch = b.depth[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
            if patch.size == 0 or np.mean((patch >= zlo) & (patch <= zhi)) < self.min_visible_frac:
                continue
            out.append(Detection(-1, o.cls, 1.0, (float(x1), float(y1), float(x2), float(y2)), source="ground_truth"))
        return out
