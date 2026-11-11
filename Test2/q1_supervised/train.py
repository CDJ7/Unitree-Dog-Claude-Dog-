#!/usr/bin/env python3
"""Q1: train the crime-scene zone classifier.

    python q1_supervised/train.py --model resnet18      # ImageNet ResNet18, fine-tuned (main model)
    python q1_supervised/train.py --model smallcnn      # small CNN from scratch (baseline)
    python q1_supervised/train.py --model resnet18 --epochs 1 --limit 500    # quick smoke test

Data: Q2 frames split by run (q1_supervised/data.py). Augmentation on the training set only:
random resized crop (scale 0.8-1, aspect ~4:3), rotation +-5 deg, colour jitter; no horizontal flip
(it would mirror the room). Each epoch uses a random `epoch_fraction` of the training frames. Loss: cross-entropy with inverse-frequency class weights and label
smoothing. Optimiser: AdamW (ResNet: lower LR on the pretrained backbone than on the new head),
cosine schedule. The checkpoint with the best validation macro-F1 is kept.

Writes results/q1/<model>/: best.pt (weights + class names + config), history.json (per-epoch
train/val loss, accuracy, macro-F1), train_config.json (every hyper-parameter, seed, split, data
counts, weights source, wall time).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

TEST2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST2))

from go2sim.zones import Zones  # noqa: E402
from q1_supervised.data import ZoneDataset, build, load_cfg, normalize  # noqa: E402
from q1_supervised.models import build_model, count_params  # noqa: E402


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def train_transform(cfg):
    from torchvision.transforms import v2 as T
    a = cfg["augment"]
    h, w = cfg["image_hw"]
    return T.Compose([
        T.RandomResizedCrop((h, w), scale=tuple(a["crop_scale"]), ratio=(w / h * 0.9, w / h * 1.1), antialias=True),
        T.RandomRotation(a["rotation_deg"]),
        T.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"]),
    ])


def macro_f1(y, p, n):
    f = []
    for c in range(n):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps, ls = [], [], []
    for x, y in loader:
        out = model(normalize(x))
        ls.append(out.float())
        ys.append(y)
    logits = torch.cat(ls)
    return torch.cat(ys).numpy(), logits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "smallcnn", "resnet18_scratch"])
    ap.add_argument("--config", help="default: $Q1_CONFIG or config/q1_train.yaml")
    ap.add_argument("--data")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="use at most N training frames (smoke tests)")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    seed_all(cfg["seed"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    names = cfg.get("classes") or Zones().names     # real-world configs list their own zone names
    nc = len(names)
    t_start = time.time()
    frames, split, cache, idx = build(cfg, a.data)
    tr_idx = idx["train"]
    if a.limit:
        tr_idx = np.random.default_rng(cfg["seed"]).choice(tr_idx, size=min(a.limit, len(tr_idx)), replace=False)
    tc = cfg["train"]
    sc = cfg.get(a.model, {}) if a.model == "smallcnn" else {}
    epochs = a.epochs or sc.get("epochs", tc["epochs"])
    g = torch.Generator().manual_seed(cfg["seed"])
    dl = dict(batch_size=tc["batch_size"], num_workers=tc["num_workers"], persistent_workers=tc["num_workers"] > 0)
    # each epoch draws a fresh random subset (epoch_fraction) of the training frames: consecutive
    # 5 Hz frames are near-duplicates, so this halves CPU time at little cost
    n_per_epoch = int(round(len(tr_idx) * tc.get("epoch_fraction", 1.0)))
    sampler = torch.utils.data.RandomSampler(range(len(tr_idx)), replacement=False, num_samples=n_per_epoch,
                                             generator=g)
    train_dl = torch.utils.data.DataLoader(ZoneDataset(cache, tr_idx, train_transform(cfg)), sampler=sampler,
                                           drop_last=True, **dl)
    val_dl = torch.utils.data.DataLoader(ZoneDataset(cache, idx["val"]), shuffle=False, **dl)

    model = build_model(a.model, nc)
    counts = np.bincount(frames.zone_id.to_numpy()[tr_idx], minlength=nc)
    w = counts.sum() / (nc * np.maximum(counts, 1)) if tc["class_weighted_loss"] else np.ones(nc)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32), label_smoothing=tc["label_smoothing"])
    if a.model.startswith("resnet18") and a.model != "resnet18_scratch":
        head = list(model.fc.parameters())
        body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        opt = torch.optim.AdamW([{"params": body, "lr": tc["lr_backbone"]}, {"params": head, "lr": tc["lr_head"]}],
                                weight_decay=tc["weight_decay"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=sc.get("lr", tc["lr_head"]), weight_decay=tc["weight_decay"])
    steps = epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[pg["lr"] for pg in opt.param_groups], total_steps=steps,
                                                pct_start=0.15, anneal_strategy="cos")
    out = Path(a.out or TEST2 / cfg["results_dir"] / a.model)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{a.model}: {count_params(model) / 1e6:.2f} M params, weights: {model.weights_source}; "
          f"train {len(tr_idx)} / val {len(idx['val'])} / test {len(idx['test'])} frames; {epochs} epochs", flush=True)
    hist, best = [], -1.0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tl, tc_, tn = time.time(), 0.0, 0, 0
        for b, (x, y) in enumerate(train_dl):
            logits = model(normalize(x))
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss.detach()) * len(y)
            tc_ += int((logits.argmax(1) == y).sum())
            tn += len(y)
            if b % 40 == 0:
                print(f"  ep {ep} batch {b}/{len(train_dl)} loss {float(loss.detach()):.3f} ({time.time() - t0:.0f} s)", flush=True)
        yv, lv = predict(model, val_dl)
        vloss = float(crit(lv, torch.from_numpy(yv)))
        pv = lv.argmax(1).numpy()
        rec = {"epoch": ep, "train_loss": tl / tn, "train_acc": tc_ / tn, "val_loss": vloss,
               "val_acc": float((pv == yv).mean()), "val_macro_f1": macro_f1(yv, pv, nc),
               "lr_backbone": opt.param_groups[0]["lr"], "epoch_s": round(time.time() - t0, 1)}
        hist.append(rec)
        tag = ""
        if rec["val_macro_f1"] > best:
            best = rec["val_macro_f1"]
            torch.save({"model": a.model, "state_dict": model.state_dict(), "classes": names,
                        "image_hw": cfg["image_hw"], "epoch": ep, "val_macro_f1": best}, out / "best.pt")
            tag = "  * best"
        print(f"epoch {ep}: train loss {rec['train_loss']:.3f} acc {rec['train_acc']:.3f} | val loss {vloss:.3f} "
              f"acc {rec['val_acc']:.3f} macro-F1 {rec['val_macro_f1']:.3f} ({rec['epoch_s']} s){tag}", flush=True)
        (out / "history.json").write_text(json.dumps(hist, indent=1))
    info = {"model": a.model, "params": count_params(model), "weights_source": model.weights_source,
            "epochs": epochs, "best_epoch": int(max(hist, key=lambda r: r["val_macro_f1"])["epoch"]),
            "best_val_macro_f1": best, "seed": cfg["seed"], "config": cfg, "split": split,
            "n_frames": {k: int(len(v)) for k, v in idx.items()}, "n_train_used": int(len(tr_idx)),
            "train_class_counts": dict(zip(names, map(int, counts))),
            "class_weights": dict(zip(names, map(float, w))), "torch": torch.__version__,
            "threads": torch.get_num_threads(), "wall_time_s": round(time.time() - t_start, 1)}
    (out / "train_config.json").write_text(json.dumps(info, indent=1))
    print(f"done in {info['wall_time_s']} s; best val macro-F1 {best:.3f} (epoch {info['best_epoch']}) -> {out}")


if __name__ == "__main__":
    main()
