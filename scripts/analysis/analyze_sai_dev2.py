#!/usr/bin/env python3
"""Comprehensive analysis of SAI dev2 extraction: response patterns by
(anchor_mode × locate × layer × σ), first-mention signals, random controls,
per-dataset breakdown — pattern discovery before score freezing."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl_records


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([neg, pos])
    sorted_idx = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[sorted_idx]
    ranks = np.empty(len(combined), float)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[sorted_idx[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_neg = len(neg)
    return float((ranks[n_neg:].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * n_neg))


def load_records(model, suffix):
    subsets = json.loads((PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json").read_text())
    pos_ids = set(subsets[model]["positive_ids"])
    neg_ids = set(subsets[model]["negative_ids"])
    records = {}
    for ds in ("vilp", "hallusionbench", "mmvet"):
        p = PROJECT_ROOT / f"results/sai/{model}/{suffix}{ds}.jsonl"
        if not p.exists():
            continue
        for obj in load_jsonl_records(p):
            if obj.get("record_type") != "sample":
                continue
            sid = obj["sample"]["sample_id"]
            records[sid] = (ds, obj["sai"])
    return records, pos_ids, neg_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava")
    parser.add_argument("--suffix", default="dev2_")
    parser.add_argument("--use", default="logit", choices=("logit", "logp"))
    args = parser.parse_args()

    records, pos_ids, neg_ids = load_records(args.model, args.suffix)
    print(f"loaded {len(records)} samples "
          f"(pos={sum(s in pos_ids for s in records)}, neg={sum(s in neg_ids for s in records)}), "
          f"signal={args.use}")

    # Collect per-sample candidate statistics per condition group.
    # groups: (anchor_mode, locate, layer, sigma)
    groups = defaultdict(lambda: defaultdict(list))  # sid_stats per group
    for sid, (ds, r) in records.items():
        lab = 1 if sid in pos_ids else 0
        by_key = {}
        for iv in r["interventions"]:
            field = "mention_dlogit" if args.use == "logit" else "mention_dlogp"
            if field not in iv:
                continue
            key = (iv["anchor_mode"], iv["locate"], iv["layer"], iv["sigma"], iv["anchor"])
            by_key.setdefault(key, {})[iv["sign"]] = iv[field]
        for (amode, loc, L, sigma, anchor), signs in by_key.items():
            if 1 not in signs or -1 not in signs:
                continue
            i = next((k for k, o in enumerate(r["objects"]) if o["surface"] == anchor), None)
            if i is None:
                continue  # random control or missing
            tw = signs[1][i]
            aw = signs[-1][i]
            if not tw or not aw:
                continue
            first = 0  # positions sorted → index 0 is the first mention
            groups[(amode, loc, L, sigma)]["first_C"].append((lab, (tw[first] - aw[first]) / 2))
            groups[(amode, loc, L, sigma)]["first_abs"].append((lab, abs((tw[first] - aw[first]) / 2)))
            groups[(amode, loc, L, sigma)]["first_dam"].append((lab, abs(tw[first]) + abs(aw[first])))
            groups[(amode, loc, L, sigma)][f"ds_{ds}"].append((lab, (tw[first] - aw[first]) / 2))
        # random control magnitudes per group with anchor_mode 'random'
        for iv in r["interventions"]:
            if iv["anchor_mode"] != "random":
                continue
            key = (iv["anchor_mode"], iv["locate"], iv["layer"], iv["sigma"])
            field = "mention_dlogit" if args.use == "logit" else "mention_dlogp"
            if field not in iv:
                continue
            arr = iv[field][0] if iv[field] else []
            if arr:
                groups[key]["rand_first_abs"].append((lab, abs(arr[0])))

    print(f"\n== First-mention response by condition (signal={args.use}; "
          f"C=(toward-away)/2, positive=belief strengthens toward anchor)")
    hdr = f"{'anchor':<14s} {'locate':<11s} {'L':>3s} {'σ':>4s} {'n':>4s} | {'AUROC(-C)':>9s} {'AUROC(C)':>8s} {'AUROC(-|d|)':>10s} | {'C_pos':>8s} {'C_neg':>8s} {'|d|_pos':>8s} {'|d|_neg':>8s} | {'rand|d|':>8s}"
    print(hdr)
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3])):
        amode, loc, L, sigma = key
        g = groups[key]
        if not g.get("first_C"):
            continue
        labels = [x[0] for x in g["first_C"]]
        C = [x[1] for x in g["first_C"]]
        ab = [x[1] for x in g["first_abs"]]
        dm = [x[1] for x in g["first_dam"]]
        pos_m = np.mean([c for l, c in zip(labels, C) if l == 1])
        neg_m = np.mean([c for l, c in zip(labels, C) if l == 0])
        dpos = np.mean([c for l, c in zip(labels, dm) if l == 1])
        dneg = np.mean([c for l, c in zip(labels, dm) if l == 0])
        rnd = np.mean([x[1] for x in g.get("rand_first_abs", [])]) if g.get("rand_first_abs") else float("nan")
        print(f"{amode:<14s} {loc:<11s} {L:3d} {sigma:4.1f} {len(C):4d} | "
              f"{1 - auroc(C, labels):9.3f} {auroc(C, labels):8.3f} {auroc([-x for x in dm], labels):10.3f} | "
              f"{pos_m:+8.4f} {neg_m:+8.4f} {dpos:8.4f} {dneg:8.4f} | {rnd:8.4f}")

    print("\n== Per-dataset AUROC(-C) for semantic conditions (composition check)")
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3])):
        amode, loc, L, sigma = key
        if amode == "random":
            continue
        g = groups[key]
        parts = []
        for ds in ("vilp", "hallusionbench", "mmvet"):
            arr = g.get(f"ds_{ds}", [])
            if len(arr) >= 6 and len(set(l for l, _ in arr)) == 2:
                parts.append(f"{ds[:4]}:{1 - auroc([v for _, v in arr], [l for l, _ in arr]):.3f}(n={len(arr)})")
        if parts:
            print(f"  {amode:<14s} {loc:<11s} L={L} σ={sigma}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
