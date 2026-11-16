"""coverage.py (where the training pictures are thin) and the recorder's survey turn. No robot or data needed."""
import json
import math
import sys
from pathlib import Path

import numpy as np

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

from real_go2.coverage import (SECTORS, Coverage, confusions, frame_poses, plan_spots, plot,  # noqa: E402
                               sector_of, write_plan, zone_weights)
from real_go2.record_real import SurveyTurn  # noqa: E402
from real_go2.zone_map import ZoneMap  # noqa: E402

CLASSES = ["lab_1", "lab_2", "open_area", "shelves"]


def fake_runs(n_runs=3, seed=0):
    """Driving tours along a 8 m corridor, always facing +x (yaw 0): zones every 2 m. Each run's odometry frame is
    rotated / shifted differently, like real sessions."""
    rng = np.random.default_rng(seed)
    runs = {}
    for r in range(n_runs):
        x = np.linspace(0.2, 7.8, 400)
        y = 1.0 + 0.3 * np.sin(x * 2 + r)
        lab = np.minimum((x // 2).astype(int), 3)
        th, t = rng.uniform(-0.5, 0.5), rng.uniform(-1, 1, 2)
        c, s = math.cos(th), math.sin(th)
        xy = np.column_stack([x, y]) @ np.array([[c, -s], [s, c]]).T + t
        runs[f"run_{r:03d}"] = (xy, np.full(len(x), th), lab, np.arange(len(x)) * 0.2)
    return runs


def test_tours_facing_one_way_cover_one_sector_and_spots_are_suggested(tmp_path):
    runs = fake_runs()
    zm, _ = ZoneMap.build(runs, CLASSES, log=lambda *a: None)
    xy, hd, zs, names = frame_poses(zm, runs)
    assert len(xy) == sum(len(r[0]) for r in runs.values())
    # every session was driven facing its own +x: in the map frame they all face the same way (one sector)
    assert len(set(sector_of(hd).tolist())) <= 2
    cov = Coverage(zm, xy, hd)
    summary, zone = cov.summary(5, CLASSES)
    assert summary["cells"] > 0 and summary["mean_sectors_covered"] <= 2.0
    metrics = {"classes": CLASSES, "confusion": [[90, 5, 5, 0], [0, 100, 0, 0], [0, 30, 60, 10], [0, 20, 0, 80]]}
    w = zone_weights(CLASSES, metrics)
    assert w[2] > w[1] and w[2] == max(w)                             # open_area is the weakest
    spots = plan_spots(cov, zone, w, 5, n_spots=4, min_sep=1.2)
    assert 1 <= len(spots) <= 4
    assert all(len(s["missing"]) >= SECTORS - 2 for s in spots)
    for a in spots:
        for b in spots:
            if a is not b:
                assert np.hypot(*(a["xy"] - b["xy"])) >= 1.2 - 1e-9
    conf = confusions(CLASSES, metrics)
    assert conf[0][:3] == ("open_area", "lab_2", 30)
    write_plan(tmp_path / "plan.md", CLASSES, summary, spots, conf, zm.info["reference"])
    text = (tmp_path / "plan.md").read_text()
    assert "Survey spots" in text and "open_area** taken for **lab_2" in text
    plot(cov, zone, CLASSES, spots, tmp_path / "cov.png", 5)
    assert (tmp_path / "cov.png").stat().st_size > 1000
    json.dumps({"spots": [{**s, "xy": list(map(float, s["xy"]))} for s in spots]})


def test_survey_turn_counts_imu_yaw_and_stops_after_a_full_turn():
    s = SurveyTurn(rate=0.5)
    s.start(0.0, (1.0, 2.0, 0.0), "open_area")
    t, yaw, out = 0.0, 0.0, 0.5
    while out is not None and t < 60:
        t += 0.1
        yaw = (yaw + 0.05 + math.pi) % (2 * math.pi) - math.pi        # IMU yaw wraps at +-pi
        out = s.step(t, yaw)
    assert out is None and s.spot["completed"] and abs(t - 2 * math.pi / 0.5) < 0.3
    assert s.spot["zone"] == "open_area" and s.spot["x"] == 1.0


def test_survey_turn_without_imu_uses_time_and_can_be_cancelled():
    s = SurveyTurn(rate=1.0)
    s.start(0.0, (math.nan, math.nan, math.nan), "lab_1")
    assert s.step(1.0, math.nan) == 1.0 and s.spot["x"] is None
    assert s.cancel() and not s.active and not s.spot["completed"]
    assert not s.cancel()
