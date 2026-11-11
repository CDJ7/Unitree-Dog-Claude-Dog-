#!/usr/bin/env python3
"""Collect the real-Go2 results into Markdown tables for REPORT.md section 7 (reads files only, no robot).

    python real_go2/summarize_real.py              # prints the tables, writes results/real_q1/summary.md

Reads, from results/real_q1/ (or real_config.yaml's results_dir):
  metrics.json      test-session accuracy, macro-F1 and per-zone scores        (evaluate_real.py)
  live_*.json       live accuracy of each live run with true-zone keys         (live_zone.py)
  goto_*.json       every go-to-zone request: target, result, reason, legs     (goto_zone.py)
and data_real/run_*/metadata.json / states.csv for the dataset size. Missing files are reported as missing.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

TEST2 = Path(__file__).resolve().parents[1]


def dataset_table(data_dir, classes):
    runs = sorted(p for p in data_dir.glob("run_*") if (p / "states.csv").exists())
    if not runs:
        return f"_No recordings in `{data_dir.relative_to(TEST2)}/`._\n"
    rows, total = [], Counter()
    for r in runs:
        with open(r / "states.csv") as fh:
            c = Counter(row["zone"] for row in csv.DictReader(fh) if row.get("zone"))
        total.update(c)
        rows.append((r.name, c))
    head = "| Session | " + " | ".join(classes) + " | Total |\n|---|" + "---|" * (len(classes) + 1) + "\n"
    body = "".join(f"| {n} | " + " | ".join(str(c.get(k, 0)) for k in classes) + f" | {sum(c.values())} |\n"
                   for n, c in rows)
    body += "| **All** | " + " | ".join(f"**{total.get(k, 0)}**" for k in classes) + f" | **{sum(total.values())}** |\n"
    return head + body


def metrics_table(out):
    p = out / "metrics.json"
    if not p.exists():
        return "_`metrics.json` missing: run `python real_go2/evaluate_real.py`._\n"
    m = json.loads(p.read_text())
    lines = [f"Test-session accuracy **{m.get('accuracy')}**, macro-F1 **{m.get('macro_f1')}**"
             + (f", 1 s smoothed accuracy {m['accuracy_smoothed_1s']}" if "accuracy_smoothed_1s" in m else "")
             + f" ({p.relative_to(TEST2)}).\n"]
    per = m.get("per_class") or {}
    if per:
        lines.append("\n| Zone | Precision | Recall | F1 | Frames |\n|---|---|---|---|---|\n")
        for k, v in per.items():
            lines.append(f"| {k} | {v.get('precision')} | {v.get('recall')} | {v.get('f1')} | {v.get('support', '')} |\n")
    return "".join(lines)


def live_table(out):
    files = sorted(out.glob("live_*.json"))
    rows = [(f.name, json.loads(f.read_text())) for f in files]
    rows = [(n, d) for n, d in rows if d.get("labelled_frames")]
    if not rows:
        return "_No live run with true-zone keys yet (`live_zone.py`, press 1-4 while driving)._\n"
    s = "| Run | Source | Labelled frames | Live accuracy |\n|---|---|---|---|\n"
    for n, d in rows:
        s += f"| {n} | {d.get('source')} | {d['labelled_frames']} | {d['live_accuracy']} |\n"
    return s


def goto_table(out):
    missions = []
    for f in sorted(out.glob("goto_*.json")):
        d = json.loads(f.read_text())
        if d.get("source", "").startswith("fake"):
            continue
        for m in d.get("missions", []):
            if m.get("result") in ("arrived", "failed"):
                missions.append({**m, "file": f.name, "lidar": d.get("lidar")})
    if not missions:
        return "_No completed go-to-zone requests on the robot yet (`goto_zone.py`)._\n"
    by = defaultdict(list)
    for m in missions:
        by[m["target"]].append(m)
    s = "| Target | Requests | Arrived | Median legs (arrived) | Failure reasons |\n|---|---|---|---|---|\n"
    for k, ms in sorted(by.items()):
        ok = [m for m in ms if m["result"] == "arrived"]
        legs = sorted(m.get("legs", 0) for m in ok)
        why = Counter(m["reason"].split("(")[0].strip() for m in ms if m["result"] == "failed")
        s += (f"| {k} | {len(ms)} | {len(ok)} | {legs[len(legs) // 2] if legs else '-'} | "
              + ("; ".join(f"{r} x{c}" for r, c in why.items()) or "-") + " |\n")
    n_ok = sum(m["result"] == "arrived" for m in missions)
    s += f"\nAll targets: **{n_ok}/{len(missions)}** arrived (cancelled and replaced requests not counted).\n"
    return s


def main():
    cfg = yaml.safe_load((TEST2 / "real_go2/real_config.yaml").read_text())
    out, data = TEST2 / cfg["results_dir"], TEST2 / cfg["data_dir"]
    text = ("## Dataset (frames per zone)\n\n" + dataset_table(data, cfg["classes"])
            + "\n## Test session\n\n" + metrics_table(out)
            + "\n## Live recognition\n\n" + live_table(out)
            + "\n## Go to zone (robot runs only)\n\n" + goto_table(out))
    print(text)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(text)
    print(f"-> {(out / 'summary.md').relative_to(TEST2)}", file=sys.stderr)


if __name__ == "__main__":
    main()
