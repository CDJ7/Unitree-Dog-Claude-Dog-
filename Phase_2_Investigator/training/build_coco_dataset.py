#!/usr/bin/env python3
"""Synthetic stain-region dataset with automatic COCO segmentation labels (A6).

    python training/build_coco_dataset.py                          # quick: 5 scenarios + 6 random, 30 images each
    python training/build_coco_dataset.py --random 40 --per-scenario 80     # full: ~3.6k images
    python training/build_coco_dataset.py --preview docs/media/a6_coco_samples.jpg

Output (RF-DETR / Roboflow COCO layout), default datasets/csi_coco/:
    train/ valid/ test/   each with images *.jpg and _annotations.coco.json

How images are made (all inside the simulator, scenario by scenario):
- camera poses on free floor at robot-camera heights (0.30-0.55 m), looking at a
  stain cluster from 0.4-3.0 m (65%), at a red distractor (10%: hard negatives)
  or anywhere (25%: background), with occlusion checks; 640x480, fovy 45-60 deg;
- lighting (headlight + room lights) and wall/floor material tints are randomised;
- labels come from MuJoCo's segmentation render: the pixels of each stain
  cluster's geoms are merged into a stain *region* (closing + hole fill), then
  polygonised; one annotation per connected region; the class is the
  mechanism of the cluster (impact, castoff, projected, drip, transfer);
- distractors are never labelled; frames with no stains are kept as negatives;
- the split is by scenario (no scenario in two splits): the five hand-made
  scenarios are the test set; of the random scenarios every fifth is valid, the rest train.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim_env  # noqa: E402,F401  (first: picks the offscreen GL backend before mujoco loads)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from sim_env.scenario_scene import PREFIX_DISTRACTOR, load_scenario_config, open_scenario  # noqa: E402
from sim_env.scenarios import scenario_paths  # noqa: E402
from sim_env.scenarios.random_sim import floor_map, write_random  # noqa: E402
from sim_env.spatter.constants import MECHANISMS  # noqa: E402

CATEGORIES = [{"id": 0, "name": "bloodstain", "supercategory": "none"}] + \
             [{"id": i + 1, "name": m, "supercategory": "bloodstain"} for i, m in enumerate(MECHANISMS)]
CAT_ID = {m: i + 1 for i, m in enumerate(MECHANISMS)}
W, H = 640, 480


def split_of(scn_id, handmade_ids, random_index=None):
    """Hand-made scenarios -> test; every 5th random scenario -> valid; the rest -> train."""
    if scn_id in handmade_ids:
        return "test"
    if random_index is not None:
        return "valid" if random_index % 5 == 4 else "train"
    h = int(hashlib.md5(scn_id.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "valid"


def cluster_geom_map(sim, gt):
    """geom id -> (cluster index, mechanism)."""
    m = sim.model
    name_to_id = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in range(m.ngeom)}
    out = {}
    mech_of_stain = {s["id"]: next(e["mechanism"] for e in gt["events"] if e["index"] == s["event"])
                     for s in gt["stains"]}
    for k, c in enumerate(gt["clusters"]):
        counts = {}
        for sid in c["stain_ids"]:
            counts[mech_of_stain.get(sid, c["mechanisms"][0])] = counts.get(mech_of_stain.get(sid, c["mechanisms"][0]), 0) + 1
        mech = max(counts, key=counts.get)
        for sid in c["stain_ids"]:
            for n, g in name_to_id.items():
                if n and (n == f"stain_{sid}" or n.startswith(f"stain_{sid}_")):
                    out[g] = (k, mech)
    return out


def randomise_look(sim, rng, base):
    m = sim.model
    amb, dif = base["headlight"]
    m.vis.headlight.ambient[:] = amb * rng.uniform(0.6, 1.4)
    m.vis.headlight.diffuse[:] = dif * rng.uniform(0.6, 1.3)
    if m.nlight:
        m.light_diffuse[:] = base["lights"] * rng.uniform(0.3, 1.5)
    for i, rgba in base["mats"].items():
        tint = rng.uniform(0.82, 1.08) * rng.uniform(0.96, 1.04, 3)     # brightness + slight colour shift
        m.mat_rgba[i, :3] = np.clip(rgba[:3] * tint, 0, 1)


def look_state(sim):
    m = sim.model
    mats = {}
    for i in range(m.nmat):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or ""
        if any(k in n for k in ("wall", "floor", "Floor")) and not n.startswith("csi_"):
            mats[i] = m.mat_rgba[i].copy()
    return {"headlight": (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy()),
            "lights": m.light_diffuse.copy() if m.nlight else None, "mats": mats}


def visible(sim, a, b, groups=np.array([1, 1, 1, 0, 0, 0], np.uint8)):
    d = np.asarray(b, float) - np.asarray(a, float)
    L = np.linalg.norm(d)
    gid = np.zeros(1, np.int32)
    t = mujoco.mj_ray(sim.model, sim.data, np.asarray(a, float), d / L, groups, 1, -1, gid)
    return t < 0 or t > L - 0.06


def sample_pose(sim, gt, scn, fm, rng, kind):
    """(camera position, forward, kind) for a view of the given kind, or None."""
    free = fm["free"] & (fm["wall_dist"] > 0.3)
    props = [np.asarray(m["position"][:2]) for m in scn.get("mannequins", []) if m.get("present", True)] + \
            [np.asarray(o["center"][:2]) for o in scn.get("obstacles", [])]
    clusters = gt["clusters"]
    if kind == "distractor" and not gt.get("distractors"):
        kind = "background"
    if kind == "stain" and clusters:
        c = clusters[rng.integers(len(clusters))]
        target = np.asarray(c["center"]) + rng.normal(0, 0.1, 3) * [1, 1, 0.5]
        n = np.asarray(c["normal"])
        d = rng.uniform(0.4, 3.0)
        if abs(n[2]) < 0.7:
            ang = np.arctan2(n[1], n[0]) + rng.uniform(-1.0, 1.0)
        else:
            ang = rng.uniform(-np.pi, np.pi)
        xy = target[:2] + d * np.array([np.cos(ang), np.sin(ang)])
    elif kind == "distractor":
        dd = gt["distractors"][rng.integers(len(gt["distractors"]))]
        target = np.asarray(dd["center"], float)
        if len(target) == 2:
            target = np.r_[target, 0.4]
        ang = rng.uniform(-np.pi, np.pi)
        xy = target[:2] + rng.uniform(0.5, 2.5) * np.array([np.cos(ang), np.sin(ang)])
    else:
        kind = "background"
        idx = np.argwhere(free)
        i, j = idx[rng.integers(len(idx))]
        xy = fm["origin"] + np.array([j, i]) * fm["res"]
        yaw, pitch = rng.uniform(-np.pi, np.pi), np.radians(rng.uniform(-30, 10))
        target = np.r_[xy, 0.4] + np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)])
    jj = int(round((xy[0] - fm["origin"][0]) / fm["res"]))
    ii = int(round((xy[1] - fm["origin"][1]) / fm["res"]))
    if not (0 <= ii < free.shape[0] and 0 <= jj < free.shape[1]) or not free[ii, jj]:
        return None
    if any(np.linalg.norm(xy - p) < 0.45 for p in props):
        return None
    cam = np.r_[xy, rng.uniform(0.30, 0.55)]
    fwd = target - cam
    if kind != "background" and not visible(sim, cam, target):
        return None
    pitch = np.degrees(np.arcsin(np.clip(fwd[2] / np.linalg.norm(fwd), -1, 1)))
    if pitch < -50 or pitch > 45:
        return None
    return cam, fwd, kind


def regions(seg, gmap, min_pixels=6):
    """[(mechanism, cluster index, polygons, bbox, area, n_stain_pixels)] for the stain regions in a frame.

    A cluster's visible stain pixels are grouped (stains closer than ~5% of the
    image diagonal belong together); each group's convex hull, padded by 2 px,
    is one region."""
    out = []
    ids = np.unique(seg[seg >= 0])
    by_cluster = {}
    for g in ids:
        if int(g) in gmap:
            k, mech = gmap[int(g)]
            by_cluster.setdefault((k, mech), []).append(int(g))
    diag = float(np.hypot(*seg.shape))
    link = int(np.clip(0.05 * diag, 12, 50))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (link, link))
    for (k, mech), gs in by_cluster.items():
        mask = np.isin(seg, gs)
        if int(mask.sum()) < min_pixels:
            continue
        grown = cv2.dilate(mask.astype(np.uint8), kernel)
        n_lab, lab = cv2.connectedComponents(grown)
        for li in range(1, n_lab):
            comp_px = mask & (lab == li)
            inner = int(comp_px.sum())
            if inner < min_pixels:
                continue
            ys, xs = np.nonzero(comp_px)
            hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
            region = np.zeros(seg.shape, np.uint8)
            cv2.fillPoly(region, [hull], 1)
            region = cv2.dilate(region, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = [cv2.approxPolyDP(c, 1.0, True).reshape(-1).astype(float).tolist()
                     for c in cnts if cv2.contourArea(c) >= 4 and len(c) >= 3]
            if not polys:
                continue
            x, y, w, h = cv2.boundingRect(region)
            out.append((mech, k, polys, [float(x), float(y), float(w), float(h)], float(region.sum()), inner))
    return out


def build(specs, out_dir, per_scenario, seed, handmade_ids, preview=None):
    rng = np.random.default_rng(seed)
    data = {s: {"images": [], "annotations": [], "categories": CATEGORIES,
                "info": {"description": "CSI synthetic bloodstain regions (Phase 2)", "version": "1.0",
                         "date_created": time.strftime("%Y-%m-%d")}} for s in ("train", "valid", "test")}
    stats = {s: {"images": 0, "annotations": 0, "negatives": 0, "per_class": {m: 0 for m in MECHANISMS}}
             for s in data}
    previews = []
    ann_id = img_id = 0
    n_random = 0
    for spec in specs:
        scn = load_scenario_config(spec)
        rand_idx = None
        if scn["id"] not in handmade_ids:
            rand_idx, n_random = n_random, n_random + 1
        split = split_of(scn["id"], handmade_ids, rand_idx)
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        sim, gt = open_scenario(scn)
        try:
            fm = floor_map(scn["environment"])
            gmap = cluster_geom_map(sim, gt)
            base = look_state(sim)
            q = sim.controller.base_qpos                     # move the robot out of the scene
            sim.data.qpos[q:q + 3] = [50.0, 50.0, 0.3]
            mujoco.mj_forward(sim.model, sim.data)
            made, tries = 0, 0
            while made < per_scenario and tries < per_scenario * 40:
                want = rng.choice(["stain", "distractor", "background"], p=[0.65, 0.10, 0.25])
                pose = None
                for _ in range(40):
                    tries += 1
                    pose = sample_pose(sim, gt, scn, fm, rng, want)
                    if pose is not None:
                        break
                if pose is None:
                    continue
                cam, fwd, kind = pose
                randomise_look(sim, rng, base)
                fovy = float(rng.choice([45.0, 52.0, 60.0]))
                frame, seg = sim.render_camera(cam, fwd, fovy=fovy, width=W, height=H, depth=False,
                                               segmentation=True)
                regs = regions(seg, gmap)
                if kind == "stain" and not regs:
                    continue
                name = f"{scn['id']}_{made:04d}.jpg"
                Image.fromarray(frame.rgb).save(out_dir / split / name, quality=92)
                data[split]["images"].append({
                    "id": img_id, "file_name": name, "width": W, "height": H, "scenario": scn["id"],
                    "camera_position": np.round(cam, 3).tolist(), "fovy": fovy, "kind": kind,
                    "K": frame.K.round(3).tolist(), "T_world_cam": frame.T_world_cam.round(5).tolist()})
                for mech, k, polys, bbox, area, npx in regs:
                    data[split]["annotations"].append({
                        "id": ann_id, "image_id": img_id, "category_id": CAT_ID[mech], "segmentation": polys,
                        "bbox": bbox, "area": area, "iscrowd": 0, "cluster": gt["clusters"][k]["id"],
                        "stain_pixels": npx})
                    stats[split]["per_class"][mech] += 1
                    ann_id += 1
                stats[split]["images"] += 1
                stats[split]["annotations"] += len(regs)
                stats[split]["negatives"] += int(not regs)
                if preview and len(previews) < 8 and regs and rng.random() < 0.4:
                    previews.append(draw(frame.rgb, regs))
                img_id += 1
                made += 1
            print(f"[dataset] {scn['id']:28s} -> {split:5s}: {made} images ({tries} poses tried)", flush=True)
        finally:
            sim.close()
    for s, d in data.items():
        if d["images"]:
            (out_dir / s / "_annotations.coco.json").write_text(json.dumps(d))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))
    if preview and previews:
        while len(previews) < 8:
            previews.append(np.zeros_like(previews[0]))
        grid = np.vstack([np.hstack(previews[:4]), np.hstack(previews[4:8])])
        Image.fromarray(grid).resize((grid.shape[1] // 2, grid.shape[0] // 2)).save(preview, quality=85)
        print(f"preview -> {preview}")
    return stats


COLORS = {"impact": (255, 60, 60), "castoff": (60, 160, 255), "projected": (255, 200, 0),
          "drip": (60, 220, 60), "transfer": (220, 60, 220)}


def draw(rgb, regs):
    img = rgb.copy()
    for mech, k, polys, bbox, area, npx in regs:
        col = COLORS[mech]
        for p in polys:
            pts = np.array(p, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, col, 2)
        x, y, w, h = map(int, bbox)
        cv2.putText(img, mech, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=int, default=6, help="number of random scenarios (train/valid)")
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--per-scenario", type=int, default=30)
    ap.add_argument("--no-handmade", action="store_true", help="skip the 5 hand-made scenarios (test set)")
    ap.add_argument("--out", default=str(ROOT / "datasets" / "csi_coco"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", help="save a grid of labelled samples here")
    a = ap.parse_args()
    out = Path(a.out)
    specs = [] if a.no_handmade else [str(i + 1) for i in range(len(scenario_paths()))]
    handmade_ids = {load_scenario_config(s)["id"] for s in specs}
    for k in range(a.random):
        specs.append(str(write_random(a.first_seed + k)))
    t0 = time.time()
    stats = build(specs, out, a.per_scenario, a.seed, handmade_ids, a.preview)
    print(json.dumps(stats, indent=1))
    print(f"dataset -> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
