#!/usr/bin/env python3
"""Offline comparison of VGS score variants from v3-component outputs.

All variants are derived from the same single forward pass per sample:
  A_vis, A_prelim, A_text (region attention masses, last-2/3 layers summed)
  D = mass-weighted normalized entropy of the answer->visual attention

Variants (higher = more uncertain):
  v1  = -A_vis / (A_vis + A_text + A_prelim)      # current VGS
  v2  = -A_vis / (A_vis + A_prelim)               # visual vs self-generation
  v3  = +D                                        # visual attention dispersion
  v4  = (1 - D) * A_vis / (A_vis + A_prelim), negated
  v5  = +A_prelim / total                         # PAS-style prelim fraction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl_records

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("hallusionbench", "vilp", "mmvet")
VARIANTS = ("v1", "v2", "v3", "v4", "v5")


def _trapz(y, x=None):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y) if x is None else fn(y, x)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([neg, pos])
    sorted_idx = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[sorted_idx]
    ranks = np.empty(len(combined))
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[sorted_idx[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_neg = len(neg)
    rank_sum_pos = ranks[n_neg:].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ls = labels[order]
    tp = np.cumsum(ls)
    fp = np.cumsum(1 - ls)
    if tp[-1] == 0 or tp[-1] == len(labels):
        return float("nan")
    precision = np.concatenate([[1.0], tp / (tp + fp)])
    recall = np.concatenate([[0.0], tp / tp[-1]])
    return float(_trapz(precision, recall))


def bootstrap_auroc_ci(scores, labels, groups, n=1000, seed=0):
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


def variant_scores(rec: dict) -> dict | None:
    v = rec.get("vgs", {})
    a_vis = v.get("visual_attn_sum")
    a_pre = v.get("prelim_attn_sum")
    a_txt = v.get("prompt_text_attn_sum")
    d = v.get("visual_attn_entropy")
    if a_vis is None or a_pre is None or a_txt is None:
        return None
    tot = a_vis + a_pre + a_txt
    if tot < 1e-12:
        return None
    out = {
        "v1": -a_vis / tot,
        "v2": -(a_vis / (a_vis + a_pre)) if (a_vis + a_pre) > 1e-12 else None,
        "v5": a_pre / tot,
    }
    if d is not None:
        out["v3"] = d
        r2 = (a_vis / (a_vis + a_pre)) if (a_vis + a_pre) > 1e-12 else 0.0
        out["v4"] = -(1.0 - d) * r2
    else:
        out["v3"] = None
        out["v4"] = None
    return out


def load_model(model: str, subset_ids: set, group_ids: dict, results_dir: Path):
    comps, judge = {}, {}
    for ds in DATASETS:
        p = results_dir / f"{model}/{ds}.jsonl"
        if not p.exists():
            continue
        for obj in load_jsonl_records(p):
            if obj.get("record_type") != "sample":
                continue
            sid = obj.get("sample", {}).get("sample_id")
            if sid not in subset_ids:
                continue
            sc = variant_scores(obj)
            if sc is not None and all(sc[v] is not None for v in VARIANTS):
                comps[sid] = sc
        p = PROJECT_ROOT / f"results/judging/{model}/{ds}.jsonl"
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") == "run":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid in subset_ids:
                    j = obj.get("judge", {})
                    if j.get("valid") is True:
                        judge[sid] = 1 if j.get("hallucination") else 0

    sids = sorted(s for s in comps if s in judge)
    labels = np.array([judge[s] for s in sids], dtype=int)
    groups = np.array([str(group_ids.get(s, s)) for s in sids], dtype=object)
    data = {v: np.array([comps[s][v] for s in sids], dtype=float) for v in VARIANTS}
    return sids, labels, groups, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/vgs_components")
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    subsets = json.loads(args.subset.read_text(encoding="utf-8"))
    group_ids = {}
    for model in MODELS:
        for ds in DATASETS:
            p = PROJECT_ROOT / f"results/generation/{model}/greedy/{ds}.jsonl"
            if p.exists():
                for obj in load_jsonl_records(p):
                    if obj.get("record_type") != "sample":
                        continue
                    sid = obj.get("sample", {}).get("sample_id")
                    if sid:
                        group_ids[sid] = obj.get("sample", {}).get("group_id", sid)

    results = {}
    print(f"{'model':10s} {'n':>4s} {'variant':8s} {'AUROC':>7s} {'CI95':>17s} {'AUPRC':>7s}")
    for model in args.models:
        sub = subsets[model]
        subset_ids = set(sub["positive_ids"]) | set(sub["negative_ids"])
        sids, labels, groups, data = load_model(model, subset_ids, group_ids, args.results_dir)
        results[model] = {"n": len(sids), "n_pos": int(labels.sum()), "metrics": {}}
        for v in VARIANTS:
            sc = data[v]
            a = auroc(sc, labels)
            lo, hi = bootstrap_auroc_ci(sc, labels, groups)
            results[model]["metrics"][v] = {"auroc": a, "auroc_ci": [lo, hi], "auprc": auprc(sc, labels)}
            print(f"{model:10s} {len(sids):4d} {v:8s} {a:7.4f} [{lo:7.4f},{hi:7.4f}] {results[model]['metrics'][v]['auprc']:7.4f}")
        print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
