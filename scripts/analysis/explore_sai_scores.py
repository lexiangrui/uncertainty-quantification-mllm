#!/usr/bin/env python3
"""Score-candidate exploration on the full development-model extraction.

Computes every candidate score per sample (coupling variants, propagation,
support, cross-specificity, combinations), then reports pooled and
per-dataset AUROC with cluster bootstrap CIs — the input to the freeze
decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl_records
from src.improvement.sai_score import parse_record, coupling, support, propagation


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


def bootstrap_ci(scores, labels, groups, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    ug = np.unique(groups)
    vals = []
    for _ in range(n):
        sel = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sel])
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(auroc(scores[idx], labels[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava")
    parser.add_argument("--subset", type=Path,
                        default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json")
    args = parser.parse_args()

    subsets = json.loads(args.subset.read_text())[args.model]
    pos_ids, neg_ids = set(subsets["positive_ids"]), set(subsets["negative_ids"])
    group_ids = {}
    records = {}
    for ds in ("vilp", "hallusionbench", "mmvet"):
        for obj in load_jsonl_records(PROJECT_ROOT / f"results/generation/{args.model}/greedy/{ds}.jsonl"):
            if obj.get("record_type") == "sample":
                group_ids[obj["sample"]["sample_id"]] = obj["sample"].get("group_id", obj["sample"]["sample_id"])
        p = PROJECT_ROOT / f"results/sai/{args.model}/{ds}.jsonl"
        if not p.exists():
            continue
        for obj in load_jsonl_records(p):
            if obj.get("record_type") == "sample":
                records[obj["sample"]["sample_id"]] = (ds, obj["sai"])

    # Build per-sample candidate scores.
    cand: dict[str, dict[str, float]] = {}
    meta: dict[str, tuple[str, str]] = {}
    for sid, (ds, sai) in records.items():
        views = parse_record(sai)
        row: dict[str, float] = {}
        for amode in ("mention_state", "unembed"):
            for loc in ("topk", "soft", "topk_large"):
                for L in (16, 24):
                    for sg in (1.0, 2.0):
                        c = coupling(views, amode, loc, L, sg)
                        if c is not None:
                            row[f"C_{amode}_{loc}_L{L}_s{sg}"] = c
        for amode in ("mention_state", "unembed"):
            for L in (16, 24):
                p31 = propagation(sai, amode, "topk", L, 1.0, 31)
                if p31 is not None:
                    row[f"P31_{amode}_L{L}_s1.0"] = p31
        s = support(views)
        if s is not None:
            row["support"] = s
            row["neg_support"] = -s
        cand[sid] = row
        meta[sid] = (ds, "pos" if sid in pos_ids else "neg")

    sids = sorted(cand)
    labels = np.array([1 if meta[s][1] == "pos" else 0 for s in sids])
    groups = np.array([str(group_ids.get(s, s)) for s in sids], dtype=object)
    datasets = np.array([meta[s][0] for s in sids])

    all_names = sorted({n for row in cand.values() for n in row})
    print(f"n={len(sids)} pos={labels.sum()}  candidates={len(all_names)}")
    print(f"{'candidate':<36s} {'pooled':>7s} {'CI95':>17s} {'vilp':>6s} {'hall':>6s} {'mmvet':>6s} {'min-ds':>7s}")

    results = []
    for name in all_names:
        vals = np.array([cand[s].get(name, np.nan) for s in sids])
        ok = np.isfinite(vals)
        if ok.sum() < 0.5 * len(sids):
            continue
        filled = np.where(ok, vals, np.nanmedian(vals))
        a = auroc(filled, labels)
        lo, hi = bootstrap_ci(filled, labels, groups)
        per = []
        for ds in ("vilp", "hallusionbench", "mmvet"):
            m = datasets == ds
            if m.sum() > 10 and len(set(labels[m])) == 2:
                per.append((ds, auroc(filled[m], labels[m]), int(m.sum())))
        min_ds = min((v for _, v, _ in per), default=float("nan"))
        print(f"{name:<36s} {a:7.3f} [{lo:6.3f},{hi:6.3f}] "
              + " ".join(f"{v:6.3f}" for _, v, _ in per)
              + f" {min_ds:7.3f}")
        results.append({"name": name, "auroc": a, "ci": [lo, hi],
                        "per_dataset": {d: v for d, v, _ in per}})

    out = PROJECT_ROOT / f"results/sai/{args.model}/score_exploration.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"output={out}")


if __name__ == "__main__":
    main()
