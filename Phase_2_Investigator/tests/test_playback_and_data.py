"""A5 event playback, A6 run recording and COCO region labels."""
import json

import cv2
import numpy as np
import pytest

import sim_env  # noqa: F401
from sim_env.runtime import SIM_ROOT

needs_sim = pytest.mark.skipif(not (SIM_ROOT / "sim/native/controller.py").exists(),
                               reason="simulator submodule not fetched")


def test_coco_regions_group_stains_and_keep_classes_apart():
    from training.build_coco_dataset import regions
    seg = np.full((480, 640), -1, np.int32)
    rng = np.random.default_rng(0)
    for x, y in rng.uniform([100, 100], [220, 200], (30, 2)).astype(int):     # cluster 0: dots
        cv2.circle(seg, (int(x), int(y)), 2, 10, -1)
    for x, y in rng.uniform([450, 300], [520, 360], (12, 2)).astype(int):     # cluster 1 far away
        cv2.circle(seg, (int(x), int(y)), 2, 20, -1)
    seg[5, 5] = 30                                                              # a 1-px speck: ignored
    gmap = {10: (0, "impact"), 20: (1, "castoff"), 30: (2, "drip")}
    regs = regions(seg, gmap)
    assert sorted(r[0] for r in regs) == ["castoff", "impact"]
    imp = next(r for r in regs if r[0] == "impact")
    x, y, w, h = imp[3]
    assert x <= 100 and y <= 100 and x + w >= 220 and y + h >= 200                # one region covers the pattern
    poly = np.array(imp[2][0]).reshape(-1, 2).astype(np.int32)
    assert cv2.pointPolygonTest(poly, (160.0, 150.0), False) >= 0


@pytest.fixture(scope="module")
def s5():
    from sim_env.scenario_scene import build_scenario, load_scenario_config
    scn = load_scenario_config(5)
    gt = json.loads((build_scenario(scn) / "ground_truth.json").read_text())
    return scn, gt


@needs_sim
def test_ground_truth_playback_lands_every_droplet_on_its_stain(s5):
    from sim_env.playback import EventPlayback
    scn, gt = s5
    pb = EventPlayback.from_ground_truth(scn, gt)
    try:
        P = pb.plan
        assert len(P.droplets) == len(gt["droplets"])
        assert set(pb.stain_geoms) <= set(P.stains_at)                 # every stain has an appear time
        impact = {d["id"]: d for d in gt["droplets"]}
        for dr, d in zip(P.droplets, gt["droplets"]):
            if d["landed"]:
                assert np.linalg.norm(dr["track"].at(dr["land"]) - impact[d["id"]]["impact_pos"]) < 1e-6
                # the re-integrated flight agrees with the generator's (same physics) just before landing
                assert np.linalg.norm(dr["track"].p[-2] - dr["track"].p[-1]) < 0.06
        assert P.swings and P.walks and P.hands                          # strike, attacker walk, smear hand
        m = pb.sim.model
        pb.set_time(0.5)                                                   # before the event: no stains
        assert all(m.geom_rgba[g, 3] == 0 for g in pb.stain_geoms)
        pb.set_time(P.duration)                                            # final scene: all stains
        assert all(m.geom_rgba[g, 3] > 0 for g in pb.stain_geoms)
    finally:
        pb.close()


@needs_sim
def test_playback_from_descriptions_synthesises_landing_droplets(s5):
    from sim_env.playback import EventPlayback, descriptions_from_ground_truth
    scn, gt = s5
    events = descriptions_from_ground_truth(gt)
    for e in events:
        e.launches = []
        e.source = "analysis"
    pb = EventPlayback.from_descriptions(scn, gt, events)
    try:
        landed = [d for d in pb.plan.droplets if d["landed"]]
        assert len(landed) > 100
        assert pb.label == "INFERRED EVENT"
    finally:
        pb.close()


@needs_sim
def test_recorder_round_trip(tmp_path):
    from navigation.explorer import Navigator
    from sim_env.recorder import RunReader, RunRecorder
    from sim_env.scenario_scene import load_scenario_config, open_scenario
    import mujoco
    sim, gt = open_scenario(load_scenario_config(5))
    try:
        sim.settle(0.5)
        rec = RunRecorder(tmp_path / "run", sim, gt)
        nav = Navigator(sim, on_tick=[rec], verbose=False)
        for _ in range(20):
            nav.tick(0, 0, 0.4)
        rec.close(nav.report())
    finally:
        sim.close()
    run = RunReader(tmp_path / "run")
    assert len(run) >= 8 and len(run.scan_index) >= 18 and len(run.states) >= 20
    assert run.ground_truth["scenario_id"] == "s5_simple_room_bat"
    f = run.frame(len(run) - 1)
    assert f.rgb.shape == (240, 320, 3) and np.isfinite(f.depth_m).mean() > 0.5
    # depth + K + pose put the centre pixel on a real surface: back-project it and check it is in the room
    p = f.backproject(159.5, 119.5)
    assert -0.1 < p[0] < 6.1 and -0.1 < p[1] < 5.1 and -0.05 < p[2] < 2.7
    scan = run.scan(0)
    assert np.isfinite(scan.ranges).sum() > 300
