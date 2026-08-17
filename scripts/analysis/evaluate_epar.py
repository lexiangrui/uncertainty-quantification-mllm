#!/usr/bin/env python3
"""Final evaluation of EPAR on LUH subsets / full test set.

EPAR — Early Prelim Attention Ratio:

    s_EPAR = A_prelim^(0..3) / (A_vis + A_prelim + A_text)^(0..3)

Attention masses from the answer-prediction rows, all heads of the first
four decoder layers, summed over answer rows. A_prelim is the mass on
already-generated tokens before the current position (scaffolding + answer
prefix, excluding the row's own position), A_vis on visual tokens, A_text
on prompt text. Higher = more hallucination risk.

Reports AUROC (group-level cluster bootstrap CI), AUPRC, PRR, ECE for
PPL / SE / UMPIRE / EPAR per model.
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
BASELINES = ("perplexity", "semantic_entropy", "umpire")
METHODS = (*BASELINES, "epar")


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


def prr(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    n = len(labels)
    total_reliable = int((1 - labels).sum())

    def area(order):
        ls = labels[order]
        k = np.arange(1, n + 1)
        cum_bad = np.cumsum(ls)
        remaining_reliable = total_reliable - (k - cum_bad)
        keep = n - k
        valid = keep > 0
        prec = remaining_reliable[valid] / keep[valid]
        return float(prec.mean())

    a_s = area(np.argsort(-scores, kind="mergesort"))
    a_rand = total_reliable / n
    a_oracle = area(np.argsort(-labels, kind="mergesort"))
    if abs(a_oracle - a_rand) < 1e-12:
        return 0.0
    return float((a_s - a_rand) / (a_oracle - a_rand))


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    n = len(labels)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        norm = np.zeros_like(scores)
    else:
        norm = (scores - lo) / (hi - lo)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for b in range(bins):
        mask = (norm >= edges[b]) & (norm < edges[b + 1])
        if b == bins - 1:
            mask = (norm >= edges[b]) & (norm <= edges[b + 1])
        if mask.sum() == 0:
            continue
        out += (mask.sum() / n) * abs(norm[mask].mean() - labels[mask].mean())
    return float(out)


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


_EPAR_LAYERS = 4  # first four decoder layers, frozen


def epar_score(v: dict) -> float | None:
    """EPAR from v5-regions components (first 4 layers).

    layer_breakdown rows are [vis, scaffold, prefix, self, text];
    prelim = scaffold + prefix (the row's own position is excluded, per
    PAS's prelim definition of strictly-previous tokens).
    """
    lb = (v or {}).get("layer_breakdown") or {}
    rows = [lb.get(str(i)) for i in range(_EPAR_LAYERS)]
    if not all(rows):
        return None
    vis = sum(r[0] for r in rows)
    pre = sum(r[1] + r[2] for r in rows)
    txt = sum(r[4] for r in rows)
    tot = vis + pre + txt
    if tot < 1e-12:
        return None
    return pre / tot


def load_all(model, subset_ids, group_ids):
    """subset_ids=None evaluates on every sample with valid judge + uq + score."""
    def wanted(sid):
        return subset_ids is None or sid in subset_ids

    epar, uq, judge = {}, {}, {}
    for ds in DATASETS:
        p = PROJECT_ROOT / f"results/epar_components/{model}/{ds}.jsonl"
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") != "sample":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if wanted(sid):
                    sc = epar_score(obj.get("epar") or obj.get("vgs"))
                    if sc is not None:
                        epar[sid] = sc
        p = PROJECT_ROOT / f"results/uq/{model}/{ds}.jsonl"
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") != "sample":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if wanted(sid):
                    row = {}
                    for m in BASELINES:
                        e = obj.get("uq", {}).get(m, {})
                        if e.get("valid") is True and e.get("score") is not None:
                            row[m] = float(e["score"])
                    if len(row) == len(BASELINES):
                        uq[sid] = row
        p = PROJECT_ROOT / f"results/judging/{model}/{ds}.jsonl"
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") == "run":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if wanted(sid):
                    j = obj.get("judge", {})
                    if j.get("valid") is True:
                        judge[sid] = 1 if j.get("hallucination") else 0

    pool = subset_ids if subset_ids is not None else set(epar) | set(uq) | set(judge)
    sids = sorted(s for s in pool if s in epar and s in uq and s in judge)
    labels = np.array([judge[s] for s in sids])
    groups = np.array([str(group_ids[s]) for s in sids], dtype=object)
    data = {}
    for m in BASELINES:
        data[m] = np.array([uq[s][m] for s in sids])
    data["epar"] = np.array([epar[s] for s in sids])
    return sids, labels, groups, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/analysis/luh/epar_final_evaluation.json")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--full-set", action="store_true",
                        help="evaluate on all judged samples instead of the LUH subsets")
    args = parser.parse_args()

    subsets = {} if args.full_set else json.loads(args.subset.read_text(encoding="utf-8"))

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
    print(f"{'model':10s} {'n':>4s} {'method':18s} {'AUROC':>7s} {'CI95':>17s} {'AUPRC':>7s} {'PRR':>7s} {'ECE':>7s}")
    for model in MODELS:
        subset_ids = None
        if not args.full_set:
            sub = subsets[model]
            subset_ids = set(sub["positive_ids"]) | set(sub["negative_ids"])
        sids, labels, groups, data = load_all(model, subset_ids, group_ids)
        results[model] = {"n": len(sids), "n_pos": int(labels.sum()), "metrics": {}}
        for m in METHODS:
            sc = data[m]
            a = auroc(sc, labels)
            lo, hi = bootstrap_auroc_ci(sc, labels, groups, args.bootstrap_samples)
            metrics = {
                "auroc": a, "auroc_ci": [lo, hi],
                "auprc": auprc(sc, labels), "prr": prr(sc, labels), "ece": ece(sc, labels),
            }
            results[model]["metrics"][m] = metrics
            print(f"{model:10s} {len(sids):4d} {m:18s} {a:7.4f} [{lo:7.4f},{hi:7.4f}] "
                  f"{metrics['auprc']:7.4f} {metrics['prr']:7.4f} {metrics['ece']:7.4f}")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
