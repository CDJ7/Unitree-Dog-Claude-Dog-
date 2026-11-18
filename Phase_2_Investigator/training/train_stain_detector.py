#!/usr/bin/env python3
"""Train and evaluate the stain candidate detector (perception/stain_detector.py), CPU only.

    python training/train_stain_detector.py                    # ~25 min on 2 threads
    python training/train_stain_detector.py --views 20         # quicker

Views are rendered like the Go2's front-camera stream: 1280 x 720, 100 deg horizontal field of view, from
0.30-0.55 m high, 0.4-3 m from a stain cluster (55 %), a red distractor (25 %) or anywhere (20 %), with
randomised lighting and wall/floor tints (the COCO builder's samplers, training/build_coco_dataset.py).
Every red region is labelled from MuJoCo's segmentation render: stain if at least half its red pixels
belong to stain geoms, else distractor / other.

Split by scene (no scene in two splits), as the COCO set: random scenarios train (every fifth one
validates, which picks the threshold), the five hand-made scenarios are the test set.
Writes perception/models/stain_rf_{depth,image}.pkl and docs/media/stage_c/detector_metrics.json
(+ detector_examples.jpg).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from perception.stain_detector import DEPTH_FEATURES, IMAGE_FEATURES, MODELS, draw, propose, region_features  # noqa: E402
from sim_env.scenario_scene import load_scenario_config, open_scenario  # noqa: E402
from sim_env.scenarios.random_sim import floor_map  # noqa: E402
from training.build_coco_dataset import cluster_geom_map, look_state, randomise_look, sample_pose  # noqa: E402

W, H, HFOV = 1280, 720, 100.0
FOVY = float(np.degrees(2 * np.arctan(np.tan(np.radians(HFOV / 2)) * H / W)))
OUT = ROOT / "docs/media/stage_c"
CACHE = ROOT / "outputs/stain_detector"


def scene_list():
    hand = [str(i) for i in range(1, 6)]
    rand = sorted((ROOT / "sim_env/scenarios/generated").glob("r3*.yaml"))
    split = {h: "test" for h in hand}
    for i, p in enumerate(rand):
        split[str(p)] = "valid" if i % 5 == 4 else "train"
    return split


def views_of(spec, n, rng, examples):
    """Feature rows [(feats dict, label, kind, view id)] and per-view cluster visibility for one scene."""
    scn = load_scenario_config(spec)
    sim, gt = open_scenario(scn)
    rows, vis = [], []
    try:
        fm = floor_map(scn["environment"])
        gmap = cluster_geom_map(sim, gt)
        m = sim.model
        dis = {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("distractor")}
        stain_ids = np.array(sorted(gmap), int)
        base = look_state(sim)
        q = sim.controller.base_qpos
        sim.data.qpos[q:q + 3] = [50.0, 50.0, 0.3]
        mujoco.mj_forward(m, sim.data)
        made = tries = 0
        while made < n and tries < n * 40:
            want = rng.choice(["stain", "distractor", "background"], p=[0.55, 0.25, 0.20])
            pose = None
            for _ in range(40):
                tries += 1
                pose = sample_pose(sim, gt, scn, fm, rng, want)
                if pose is not None:
                    break
            if pose is None:
                continue
            randomise_look(sim, rng, base)
            frame, seg = sim.render_camera(pose[0], pose[1], fovy=FOVY, width=W, height=H, depth=True, segmentation=True)
            regs, hsv = propose(frame.rgb)
            region_features(frame.rgb, regs, hsv, frame)
            vid = f"{scn['id']}#{made}"
            for r in regs:
                x, y, w, h = r.bbox
                ids = seg[y:y + h, x:x + w][r.mask]
                sf = float(np.isin(ids, stain_ids).mean())
                df = float(np.isin(ids, list(dis)).mean())
                kind = "stain" if sf >= 0.5 else ("distractor" if df >= 0.5 else "other")
                clusters = sorted({gmap[int(g)][0] for g in ids if int(g) in gmap})
                rows.append((r.feats, int(kind == "stain"), kind, vid, clusters))
            # clusters visible in this view: >= 15 stain pixels in the segmentation
            visible = {}
            for g, (k, _) in gmap.items():
                c = int((seg == g).sum())
                if c:
                    visible[k] = visible.get(k, 0) + c
            vis.append((vid, [k for k, c in visible.items() if c >= 15]))
            if examples is not None and len(examples) < 6 and any(rw[2] == "stain" for rw in rows[-len(regs):] if regs) \
                    and rng.random() < 0.3:
                examples.append((frame, regs, vid))
            made += 1
    finally:
        sim.close()
    return rows, vis


def matrix(rows, feats):
    return np.array([[r[0].get(f, 0.0) for f in feats] for r in rows], float), np.array([r[1] for r in rows])


def evaluate(clf, rows, vis, feats, thr):
    X, y = matrix(rows, feats)
    p = clf.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
    pred = p >= thr
    kinds = np.array([r[2] for r in rows])
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    out = {"regions": int(len(y)), "stain_regions": int(y.sum()),
           "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
           "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
           "distractor_regions": int((kinds == "distractor").sum()),
           "distractor_false_positive_rate": round(float(pred[kinds == "distractor"].mean()), 4)
           if (kinds == "distractor").any() else None,
           "other_false_positive_rate": round(float(pred[kinds == "other"].mean()), 4) if (kinds == "other").any() else None}
    # cluster level: a cluster visible in a view counts as found if a region predicted "stain" covers it
    found = {}
    for (f, lab, kind, vid, cl), pr in zip(rows, pred):
        if pr:
            for k in cl:
                found.setdefault(vid, set()).add(k)
    n_vis = sum(len(v) for _, v in vis)
    n_found = sum(len(set(v) & found.get(vid, set())) for vid, v in vis)
    out["cluster_views"] = n_vis
    out["cluster_recall_per_view"] = round(n_found / max(n_vis, 1), 4)
    fp_views = {r[3] for r, pr in zip(rows, pred) if pr and r[1] == 0}
    out["views"] = len(vis)
    out["views_with_a_false_detection"] = round(len(fp_views) / max(len(vis), 1), 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", type=int, default=40, help="views per scene")
    ap.add_argument("--seed", type=int, default=6010)
    ap.add_argument("--reuse", action="store_true", help="reuse the cached feature rows")
    a = ap.parse_args()
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(a.seed)
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"rows_{a.views}.pkl"
    split = scene_list()
    examples = []
    if a.reuse and cache.exists():
        data = pickle.loads(cache.read_bytes())
    else:
        data = {"train": ([], []), "valid": ([], []), "test": ([], [])}
        t0 = time.time()
        for spec, sp in split.items():
            rows, vis = views_of(spec, a.views, rng, examples if sp == "test" else None)
            data[sp][0].extend(rows)
            data[sp][1].extend(vis)
            print(f"[detector] {Path(spec).stem:6s} {sp:5s}: {len(vis)} views, {len(rows)} regions "
                  f"({sum(r[1] for r in rows)} stain)  {time.time() - t0:5.0f}s", flush=True)
        cache.write_bytes(pickle.dumps(data))
    metrics = {"camera": {"width": W, "height": H, "hfov_deg": HFOV}, "views_per_scene": a.views,
               "scenes": {sp: sorted({Path(s).stem for s, v in split.items() if v == sp}) for sp in ("train", "valid", "test")}}
    for name, feats in (("depth", IMAGE_FEATURES + DEPTH_FEATURES), ("image", IMAGE_FEATURES)):
        Xtr, ytr = matrix(data["train"][0], feats)
        clf = RandomForestClassifier(n_estimators=150, max_depth=14, min_samples_leaf=2, class_weight="balanced",
                                     n_jobs=int(sim_env.THREADS), random_state=a.seed).fit(Xtr, ytr)
        # threshold: best F1 on the validation scenes
        Xv, yv = matrix(data["valid"][0], feats)
        pv = clf.predict_proba(Xv)[:, 1]
        best = max(np.arange(0.2, 0.91, 0.05), key=lambda t: 2 * ((pv >= t) & (yv == 1)).sum() /
                   max(((pv >= t).sum() + (yv == 1).sum()), 1))
        thr = round(float(best), 2)
        res = {"threshold": thr, "train_regions": int(len(ytr)), "train_stain_regions": int(ytr.sum()),
               "valid": evaluate(clf, data["valid"][0], data["valid"][1], feats, thr),
               "test": evaluate(clf, data["test"][0], data["test"][1], feats, thr),
               "importance": dict(sorted(((f, round(float(v), 4)) for f, v in zip(feats, clf.feature_importances_)),
                                         key=lambda kv: -kv[1])[:8])}
        metrics[name] = res
        (MODELS / f"stain_rf_{name}.pkl").write_bytes(pickle.dumps({"model": clf, "features": feats, "threshold": thr}))
        t = res["test"]
        print(f"[detector] {name:5s} model: threshold {thr}; test precision {t['precision']} recall {t['recall']} "
              f"F1 {t['f1']}; distractor FP rate {t['distractor_false_positive_rate']}; "
              f"cluster recall per view {t['cluster_recall_per_view']}", flush=True)
    (OUT / "detector_metrics.json").write_text(json.dumps(metrics, indent=1))
    if examples:
        clf_blob = pickle.loads((MODELS / "stain_rf_depth.pkl").read_bytes())
        tiles = []
        for frame, regs, vid in examples[:6]:
            X = np.array([[r.feats.get(f, 0.0) for f in clf_blob["features"]] for r in regs], float)
            for r, p in zip(regs, clf_blob["model"].predict_proba(X)[:, 1]):
                r.prob = float(p)
            img = draw(frame.rgb, regs, clf_blob["threshold"])
            cv2.putText(img, vid, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (640, 360)))
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(OUT / "detector_examples.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"-> {OUT / 'detector_metrics.json'}")


if __name__ == "__main__":
    main()
