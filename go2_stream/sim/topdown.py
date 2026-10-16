"""Top-down OpenCV map of the sim room: objects, robot pose + camera FOV, trail, optional estimates."""
import math

import cv2
import numpy as np

# class -> BGR colour
_COLORS = {"chair": (200, 120, 40), "dining table": (60, 110, 160), "backpack": (60, 60, 200),
           "bottle": (80, 180, 80), "cup": (220, 220, 220), "suitcase": (70, 140, 190), "person": (200, 60, 200)}


class TopDownMap:
    def __init__(self, world, x_range=(-3.1, 3.1), y_range=(-2.6, 2.6), px_per_m=100):
        self.world, self.xr, self.yr, self.s = world, x_range, y_range, px_per_m
        self.w = int((x_range[1] - x_range[0]) * px_per_m)
        self.h = int((y_range[1] - y_range[0]) * px_per_m)

    def _px(self, x, y):
        return int((x - self.xr[0]) * self.s), int((self.yr[1] - y) * self.s)

    def draw(self, estimates=None):
        """estimates: optional list of (class, x, y) world positions to draw as hollow circles."""
        img = np.full((self.h, self.w, 3), 245, np.uint8)
        for o in self.world.objects:
            p1, p2 = self._px(o.aabb_min[0], o.aabb_max[1]), self._px(o.aabb_max[0], o.aabb_min[1])
            col = _COLORS.get(o.cls, (120, 120, 120))
            cv2.rectangle(img, p1, p2, col, -1)
            cv2.putText(img, o.cls, (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
        trail = self.world.trail
        for a, b in zip(trail, trail[1:]):
            cv2.line(img, self._px(*a), self._px(*b), (0, 140, 255), 1)
        x, y, yaw = self.world.pose
        half = math.radians(self.world.cfg["camera"]["hfov_deg"]) / 2
        tip = self._px(x, y)
        pts = [tip] + [self._px(x + 1.2 * math.cos(yaw + a), y + 1.2 * math.sin(yaw + a)) for a in (-half, half)]
        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(pts, np.int32)], (0, 200, 255))
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.circle(img, tip, 7, (0, 0, 255), -1)
        cv2.line(img, tip, self._px(x + 0.3 * math.cos(yaw), y + 0.3 * math.sin(yaw)), (255, 255, 255), 2)
        for cls, ex, ey in estimates or []:
            cv2.circle(img, self._px(ex, ey), 9, (0, 0, 0), 2)
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), (80, 80, 80), 2)
        state = "standing" if self.world.standing else "lying (press R)"
        cv2.putText(img, f"x {x:+.2f} y {y:+.2f} yaw {(math.degrees(yaw) + 180) % 360 - 180:+.0f}  {state}", (8, self.h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        return img
