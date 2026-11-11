"""Q1 data: index the Q2 frames, split them BY RUN, and serve them as tensors.

Frames are the rows of each run's states.csv with frame >= 0; the label is zone_id (the zone that
contains the dog's base when the frame was rendered). Frames outside every zone (doorway gaps, < 1%)
are dropped.

Split (config/q1_train.yaml): whole runs only, so no frame (and no near-duplicate neighbour frame
at 5 Hz) of a test route is ever seen in training:
- test: every run of the held-out scene(s) + one route from each other scene;
- val: one other route from each remaining scene;
- train: the rest.

`FrameCache` decodes every JPEG once, resized to image_hw, into one uint8 tensor (about 2 GB for the
full set) so that CPU training is limited by the network, not by JPEG decoding.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

TEST2 = Path(__file__).resolve().parents[1]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_cfg(path=None):
    """The Q1 config: `path`, else $Q1_CONFIG (e.g. config/q1_sample.yaml in the Colab quick check), else
    config/q1_train.yaml. Relative paths are resolved against Test2/."""
    path = Path(path or os.environ.get("Q1_CONFIG") or "config/q1_train.yaml")
    return yaml.safe_load((path if path.is_absolute() else TEST2 / path).read_text())


def index_frames(data_dir):
    """DataFrame: run, scene, route, frame, t, zone_id, path, base_x, base_y, base_yaw."""
    rows = []
    for meta_p in sorted(Path(data_dir).glob("run_*/metadata.json")):
        m = json.loads(meta_p.read_text())
        d = pd.read_csv(meta_p.parent / "states.csv", usecols=["t", "frame", "zone_id", "base_x", "base_y", "base_yaw"])
        d = d[(d.frame >= 0) & (d.zone_id >= 0)].copy()
        d["run"], d["scene"], d["route"] = m["run"], m["scene"], m["route"]["name"]
        d["path"] = [str(meta_p.parent / "images" / f"{f:06d}.jpg") for f in d.frame]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def split_runs(frames, cfg, collect_cfg_path=TEST2 / "config/collect.yaml"):
    """{'train': [runs], 'val': [...], 'test': [...]} from the rules in cfg['split']."""
    sp = cfg["split"]
    if sp.get("mode") == "by_order":        # real-world recordings: last session test, second-to-last val
        runs = sorted(frames.run.unique())
        assert len(runs) >= 3, f"need at least 3 recorded sessions for train/val/test, found {len(runs)}"
        return {"train": runs[:-2], "val": [runs[-2]], "test": [runs[-1]]}
    scenes = [s["name"] for s in yaml.safe_load(Path(collect_cfg_path).read_text())["scenes"]]
    runs = frames.groupby("run").agg(scene=("scene", "first"), route=("route", "first")).reset_index()
    out = {"train": [], "val": [], "test": []}
    for i, sc in enumerate(s for s in scenes if s in set(runs.scene)):
        r = runs[runs.scene == sc].sort_values("run")
        names = list(r.run)
        if sc in sp["test_scenes"]:
            out["test"] += names
            continue
        k = len(names)
        t_i, v_i = (i + sp["test_route_offset"]) % k, (i + sp["val_route_offset"]) % k
        for j, n in enumerate(names):
            out["test" if j == t_i else "val" if j == v_i else "train"].append(n)
    assert not (set(out["train"]) & set(out["val"]) or set(out["train"]) & set(out["test"])
                or set(out["val"]) & set(out["test"])), "run leakage between splits"
    return out


class FrameCache:
    """All frames decoded once into a uint8 tensor N x 3 x H x W (cached to disk as .npy)."""

    def __init__(self, frames, hw, cache_path=None, threads=8):
        self.frames = frames.reset_index(drop=True)
        self.hw = tuple(hw)
        cache_path = Path(cache_path) if cache_path else None
        key = f"{len(frames)}_{hw[0]}x{hw[1]}_" + hashlib.md5("\n".join(self.frames.path).encode()).hexdigest()
        if cache_path is not None and cache_path.with_suffix(".key").exists() and \
                cache_path.with_suffix(".key").read_text() == key and cache_path.exists():
            self.images = torch.from_numpy(np.load(cache_path, mmap_mode=None))
            return
        h, w = self.hw
        arr = np.empty((len(self.frames), 3, h, w), np.uint8)

        def load(i):
            im = Image.open(self.frames.path[i]).convert("RGB").resize((w, h), Image.BILINEAR)
            arr[i] = np.asarray(im).transpose(2, 0, 1)

        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(load, range(len(self.frames))))
        self.images = torch.from_numpy(arr)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, arr)
            cache_path.with_suffix(".key").write_text(key)


class ZoneDataset(torch.utils.data.Dataset):
    """Indices into a FrameCache; returns (float image 3xHxW in [0,1], label)."""

    def __init__(self, cache, idx, transform=None):
        self.cache, self.idx, self.transform = cache, np.asarray(idx), transform
        self.labels = cache.frames.zone_id.to_numpy()[self.idx]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        x = self.cache.images[i].float() / 255.0
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[k])


def normalize(x):
    """ImageNet normalisation of a batch in [0,1] (done on the batch, after augmentation)."""
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def build(cfg, data_dir=None, cache=True):
    """(frames DataFrame, split dict, FrameCache, {split: index array})."""
    data_dir = Path(data_dir or TEST2 / cfg["data_dir"])
    frames = index_frames(data_dir)
    split = split_runs(frames, cfg)
    fc = FrameCache(frames, cfg["image_hw"], data_dir / "_q1_cache" / "frames.npy" if cache else None)
    idx = {k: np.flatnonzero(frames.run.isin(v).to_numpy()) for k, v in split.items()}
    return frames, split, fc, idx
