"""Zone lookup: which crime-scene zone contains a point (config/zones.yaml)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from go2sim.paths import CONFIG


class Zones:
    def __init__(self, path: str | Path = CONFIG / "zones.yaml"):
        cfg = yaml.safe_load(Path(path).read_text())
        self.environment = cfg["environment"]
        self.items = cfg["zones"]
        self.names = [z["name"] for z in self.items]
        self.colors = [z.get("color", "#888888") for z in self.items]
        self.polygons = [np.asarray(z["polygon"], float) for z in self.items]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _inside(poly, x, y):
        """Even-odd ray casting; points on the min edges count as inside."""
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < xc:
                    inside = not inside
        return inside

    def zone_id(self, xy) -> int:
        """Zone id containing xy, or -1 if outside all zones."""
        x, y = float(xy[0]), float(xy[1])
        for i, poly in enumerate(self.polygons):
            if self._inside(poly, x, y):
                return i
        return -1

    def zone_name(self, xy) -> str:
        i = self.zone_id(xy)
        return self.names[i] if i >= 0 else "outside"

    def centroid(self, i):
        return self.polygons[i].mean(axis=0)
