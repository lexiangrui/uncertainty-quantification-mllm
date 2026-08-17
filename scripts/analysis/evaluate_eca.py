#!/usr/bin/env python3
"""Evaluation of ECA on the LUH subsets.

Layer band is passed explicitly; it is specified in relative depth so it
maps to each model's layer count.  Per model the per-layer feature values
are averaged inside the band (U_ECA = mean_l U_ECA^(l)).  Bands chosen
from the LUH sweep are in-sample for the band choice — label as such.

Methods compared on identical samples and judge labels:
  PPL / SE / UMPIRE (baselines), GCAR (predecessor, from epar_components),
  and the ECA feature chain at the locked band: aAI, U_direct, G_V, G_R,
  U_ECA.
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
EPS = 1e-8


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
    norm = np.zeros_like(scores) if hi - lo < 1e-12 else (scores - lo) / (hi - lo)
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


def layer_features(e: dict) -> dict[int, dict[str, float]]:
    H = e["n_heads"]
    st = e["section_tokens"]
    sizes = [st["vision"], st["reasoning"], st["answer"]]
    out = {}
    for l, m in e["layer_masses"].items():
        a = [[m[g][b] / (H * sizes[g]) for b in range(5)] for g in range(3)]
        aVI, aVQ = a[0][0], a[0][1]
        aRI, aRQ, aRV = a[1][0], a[1][1], a[1][2]
        aAI, aAQ, aAV, aAR = a[2][0], a[2][1], a[2][2], a[2][3]
        G_V = aVI / (aVI + aVQ + EPS)
        G_R = (aRI + aRV * G_V) / (aRI + aRQ + aRV + EPS)
        G_A = (aAI + aAV * G_V + aAR * G_R) / (aAI + aAQ + aAV + aAR + EPS)
        out[int(l)] = {
            "aAI": aAI, "aAQ": aAQ, "aAV": aAV, "aAR": aAR,
            "U_direct": (aAV + aAR) / (aAV + aAR + aAI + aAQ + EPS),
            "G_V": G_V, "G_R": G_R,
            "U_ECA": 1.0 - G_A,
        }
    return out


def band_mean(feats: dict[int, dict[str, float]], band: tuple[float, float]) -> dict[str, float]:
    L = max(feats) + 1
    ls = [l for l in sorted(feats) if band[0] <= l / (L - 1) <= band[1]]
    if not ls:
        ls = [max(feats)]
    return {k: float(np.mean([feats[l][k] for l in ls])) for k in feats[ls[0]]}


def gcar_score(v: dict) -> float | None:
    lb = (v or {}).get("layer_breakdown") or {}
    rows = [lb.get(str(i)) for i in range(4)]
    if not all(rows):
        return None
    num = sum(r[1] + r[2] for r in rows)
    tot = sum(sum(r) for r in rows)
    return num / tot if tot > 1e-12 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--band-rel", nargs=2, type=float, required=True, metavar=("LO", "HI"),
                        help="locked layer band as relative depth, e.g. 0.33 1.0")
    parser.add_argument("--features", nargs="+", default=["aAI", "U_direct", "G_V", "G_R", "U_ECA"])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/analysis/eca/final_evaluation.json")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    band = tuple(args.band_rel)

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
    print(f"locked band (relative depth): [{band[0]}, {band[1]}]")
    print(f"{'model':10s} {'n':>4s} {'method':18s} {'AUROC':>7s} {'CI95':>17s} {'AUPRC':>7s} {'PRR':>7s} {'ECE':>7s}")
    for model in args.models:
        sub = subsets[model]
        subset_ids = set(sub["positive_ids"]) | set(sub["negative_ids"])
        feats, gcar, uq, judge = {}, {}, {}, {}
        for ds in DATASETS:
            p = PROJECT_ROOT / f"results/eca_components/{model}/{ds}.jsonl"
            if p.exists():
                for obj in load_jsonl_records(p):
                    if obj.get("record_type") != "sample":
                        continue
                    sid = obj.get("sample", {}).get("sample_id")
                    if sid in subset_ids:
                        feats[sid] = band_mean(layer_features(obj["eca"]), band)
            p = PROJECT_ROOT / f"results/epar_components/{model}/{ds}.jsonl"
            if p.exists():
                for obj in load_jsonl_records(p):
                    if obj.get("record_type") != "sample":
                        continue
                    sid = obj.get("sample", {}).get("sample_id")
                    if sid in subset_ids:
                        sc = gcar_score(obj.get("epar") or obj.get("vgs"))
                        if sc is not None:
                            gcar[sid] = sc
            p = PROJECT_ROOT / f"results/uq/{model}/{ds}.jsonl"
            if p.exists():
                for obj in load_jsonl_records(p):
                    if obj.get("record_type") != "sample":
                        continue
                    sid = obj.get("sample", {}).get("sample_id")
                    if sid in subset_ids:
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
                    if sid in subset_ids:
                        j = obj.get("judge", {})
                        if j.get("valid") is True:
                            judge[sid] = 1 if j.get("hallucination") else 0

        sids = sorted(s for s in subset_ids if s in feats and s in uq and s in judge)
        labels = np.array([judge[s] for s in sids])
        groups = np.array([str(group_ids.get(s, s)) for s in sids], dtype=object)
        data = {m: np.array([uq[s][m] for s in sids]) for m in BASELINES}
        for f in args.features:
            data[f] = np.array([feats[s][f] for s in sids])
        if all(s in gcar for s in sids):
            data["gcar"] = np.array([gcar[s] for s in sids])

        results[model] = {"n": len(sids), "n_pos": int(labels.sum()), "metrics": {}}
        for m, sc in data.items():
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
