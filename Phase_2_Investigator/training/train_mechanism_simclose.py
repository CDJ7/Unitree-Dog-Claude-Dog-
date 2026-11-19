#!/usr/bin/env python3
"""Train the mechanism classifier on stains measured from *simulator close-ups* (C5).

    python training/train_mechanism_simclose.py              # ~40-60 min on 2 threads
    python training/train_mechanism_simclose.py --reuse      # re-train from the cached groups

Why: Phase 2's classifier (training/train_mechanism_classifier.py) learns from ideal synthetic measurements
of every stain. From the dog's own close-ups it called both scenario-1 groups "projected" (impact and cast-off
in truth), because close-ups miss the small round stains at a pattern's edges and see the rest from 0.5-1.6 m.
Here the training groups go through exactly what the mission does:
- close-ups placed like navigation/closeups.py: aim points across a stain cluster, the camera at the dog's
  height (0.40-0.50 m), 0.5-1.2 m from wall stains / 0.4-0.8 m from floor stains, pitch clamped to 50 deg,
  line of sight checked; each scene is captured in `variants` versions covering 50-100 % of the aims;
- every red blob measured by analysis/closeup.py and merged across frames, grouped as analysis/scene_report.py
  groups them (all stains, >= 3 usable);
- a group's label is the mechanism of the event most of its matched stains belong to (purity >= 0.7).
Split by scene: random scenes train (every fifth validates), the five hand-made scenarios are the test set.
Three models are compared on validation: Phase 2's original (ideal synthetic), close-up only, and close-up +
ideal synthetic groups. The best on validation is saved as perception/models/mechanism_gb_simclose.pkl, and
all three are reported on the test scenes in docs/media/stage_c/mechanism_simclose.json.
"""
import argparse
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from analysis.bpa.measure import group_observations  # noqa: E402
from analysis.classifier.features import FEATURE_NAMES, group_features  # noqa: E402
from analysis.classifier.model import DEFAULT_MODEL, MechanismClassifier  # noqa: E402
from analysis.closeup import merge_observations, stains_in_frame  # noqa: E402
from navigation.closeups import aim_points  # noqa: E402
from sim_env.scenario_scene import load_scenario_config, open_scenario  # noqa: E402

OUT_MODEL = ROOT / "perception/models/mechanism_gb_simclose.pkl"
OUT_JSON = ROOT / "docs/media/stage_c/mechanism_simclose.json"
CACHE = ROOT / "outputs/mechanism_simclose"
MECHS = ["impact", "castoff", "projected", "drip"]
ENVS = ["scenesmith_house_185", "scenesmith_house_186", "csi_simple_room"]


def scene_list(n_new):
    from sim_env.scenarios.random_sim import write_random
    gen = ROOT / "sim_env/scenarios/generated"
    rand = [str(p) for p in sorted(gen.glob("r3[0-2]*.yaml"))]
    for k in range(n_new):                       # extra scenes: all mechanisms, all three environments
        seed = 3401 + k
        p = gen / f"r{seed}.yaml"
        if not p.exists():
            write_random(seed, env=ENVS[k % 3], mechanism=MECHS[(k // 3) % 4])
        rand.append(str(p))
    split = {str(i): "test" for i in range(1, 6)}
    for i, p in enumerate(rand):
        split[p] = "valid" if i % 5 == 4 else "train"
    return split


def visible(sim, a, b):
    d = np.asarray(b, float) - np.asarray(a, float)
    L = np.linalg.norm(d)
    gid = np.zeros(1, np.int32)
    t = mujoco.mj_ray(sim.model, sim.data, np.asarray(a, float), d / L, np.array([1, 1, 1, 0, 0, 0], np.uint8), 1, -1, gid)
    return t < 0 or t > L - 0.03


def camera_for(sim, aim, normal, rng, tries=20):
    n = np.asarray(normal, float)
    for _ in range(tries):
        if abs(n[2]) > 0.8:
            ang = rng.uniform(-np.pi, np.pi)
            u = np.array([np.cos(ang), np.sin(ang)])
            h = rng.uniform(0.4, 0.8)
        else:
            base = n[:2] / max(np.linalg.norm(n[:2]), 1e-9)
            ang = rng.uniform(-0.5, 0.5)
            u = np.array([np.cos(ang) * base[0] - np.sin(ang) * base[1], np.sin(ang) * base[0] + np.cos(ang) * base[1]])
            h = rng.uniform(0.5, 1.2)
        cam = np.r_[aim[:2] + u * h, rng.uniform(0.40, 0.50)]
        if visible(sim, cam, aim):
            return cam
    return None


def shoot(sim, cam, aim, max_pitch=50.0):
    d = aim - cam
    pitch = np.clip(np.arctan2(d[2], np.hypot(d[0], d[1])), -np.radians(max_pitch), np.radians(max_pitch))
    yaw = np.arctan2(d[1], d[0])
    fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    return sim.render_camera(cam, fwd, fovy=40.0, width=1920, height=1080, depth=True)


def label(group, gt, P, st, tol=0.006):
    ev = []
    for o in group:
        d = np.linalg.norm(P - o.position, axis=1)
        k = int(np.argmin(d))
        if d[k] <= max(tol, 0.6 * st[k]["length"]):
            ev.append(st[k]["event"])
    if len(ev) < 3:
        return None, 0.0
    e, c = Counter(ev).most_common(1)[0]
    return gt["events"][e]["mechanism"], c / len(ev)


def scene_groups(spec, rng, variants=3):
    """(features dict, label, purity, n) for the close-up groups of one scene."""
    scn = load_scenario_config(spec)
    sim, gt = open_scenario(scn)
    out = []
    try:
        q = sim.controller.base_qpos
        sim.data.qpos[q:q + 3] = [50.0, 50.0, 0.3]
        mujoco.mj_forward(sim.model, sim.data)
        st = [s for s in gt["stains"]]
        P = np.array([s["center_world"] for s in st])
        by_id = {s["id"]: s for s in st}
        for v in range(variants):
            obs = []
            for c in gt["clusters"]:
                if c["n_stains"] < 4:
                    continue
                pts = np.array([by_id[i]["center_world"] for i in c["stain_ids"] if i in by_id])
                aims = aim_points(pts, radius=0.30, max_aims=12)
                keep = rng.uniform(0.5, 1.0)
                aims = [a for a in aims if rng.random() < keep] or aims[:1]
                for k, a in enumerate(aims):
                    cam = camera_for(sim, a, c["normal"], rng)
                    if cam is None:
                        continue
                    fr = shoot(sim, cam, np.asarray(a, float))
                    obs += stains_in_frame(fr, frame_id=f"{c['id']}_{v}_{k}")
            obs = merge_observations(obs)
            for g in group_observations(obs, link=0.3):
                if sum(o.usable for o in g) < 3:
                    continue
                mech, pur = label([o for o in g if o.usable], gt, P, st)
                if mech is None:
                    continue
                out.append((group_features(g), mech, pur, len(g)))
    finally:
        sim.close()
    return out


def xy(groups, min_purity=0.7):
    g = [x for x in groups if x[2] >= min_purity]
    return np.array([[f[k] for k in FEATURE_NAMES] for f, *_ in g]), np.array([m for _, m, *_ in g])


def accuracy(clf, X, y):
    if not len(y):
        return None
    pred = np.array([max(p, key=p.get) for p in (clf.predict_proba_features(dict(zip(FEATURE_NAMES, r))) for r in X)])
    return round(float((pred == y).mean()), 3), pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-scenes", type=int, default=36, help="extra random scenes (all mechanisms and houses)")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=6010)
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "groups.pkl"
    split = scene_list(a.new_scenes)
    if a.reuse and cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        data = {"train": [], "valid": [], "test": [], "per_scene": {}}
        t0 = time.time()
        for spec, sp in split.items():
            try:
                g = scene_groups(spec, rng, a.variants)
            except Exception as e:                           # a scene that fails to build is skipped, and said so
                print(f"[simclose] {Path(spec).stem}: skipped ({e!r})"[:200], flush=True)
                continue
            data[sp] += g
            data["per_scene"][Path(spec).stem] = Counter(m for _, m, *_ in g)
            print(f"[simclose] {Path(spec).stem:22s} {sp:5s}: {len(g)} groups {dict(Counter(m for _, m, *_ in g))} "
                  f"{time.time() - t0:5.0f}s", flush=True)
        cache.write_bytes(pickle.dumps(data))
    Xtr, ytr = xy(data["train"])
    Xva, yva = xy(data["valid"])
    Xte, yte = xy(data["test"])
    print(f"[simclose] groups: train {len(ytr)} {dict(Counter(ytr))}, valid {len(yva)}, test {len(yte)} {dict(Counter(yte))}")
    models = {"ideal_synthetic (Phase 2)": MechanismClassifier.load(DEFAULT_MODEL)}
    models["simclose"] = MechanismClassifier.train(Xtr, ytr, seed=a.seed)
    # close-up groups + Phase 2's ideal synthetic groups (rebuilt the same way as its training script)
    import training.train_mechanism_classifier as T
    seeds = np.arange(0, 400)
    Xs, ys, _ = T.build(seeds, 1)
    models["simclose + ideal_synthetic"] = MechanismClassifier.train(np.vstack([Xtr, Xs]), np.r_[ytr, ys], seed=a.seed)
    rep = {"groups": {"train": int(len(ytr)), "valid": int(len(yva)), "test": int(len(yte))},
           "train_labels": dict(Counter(ytr.tolist())), "test_labels": dict(Counter(yte.tolist())), "models": {}}
    for name, clf in models.items():
        va = accuracy(clf, Xva, yva)
        te = accuracy(clf, Xte, yte)
        classes = sorted(set(yte.tolist()) | set(te[1].tolist())) if te else []
        conf = [[int(((yte == t) & (te[1] == p)).sum()) for p in classes] for t in classes] if te else []
        rep["models"][name] = {"valid_accuracy": va[0] if va else None, "test_accuracy": te[0] if te else None,
                               "test_confusion": {"classes": classes, "rows_true_cols_pred": conf}}
        print(f"[simclose] {name:28s} valid {va[0] if va else None}  test {te[0] if te else None}")
    best = max((n for n in models if n != "ideal_synthetic (Phase 2)"),
               key=lambda n: rep["models"][n]["valid_accuracy"] or 0)
    rep["chosen_on_validation"] = best
    models[best].save(OUT_MODEL)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=1))
    print(f"[simclose] chosen on validation: {best} -> {OUT_MODEL.relative_to(ROOT)}; report {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
