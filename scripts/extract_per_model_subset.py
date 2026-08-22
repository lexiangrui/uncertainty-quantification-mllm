#!/usr/bin/env python3
"""Extract per-model LUH subsets with nearest-neighbour matching.

For each model independently:
1. Merge 3 datasets, compute per-method average-rank percentile.
2. Positive (LUH): H=1 samples with lowest average percentile (200).
3. Negative: for each positive, greedily find the nearest H=0 sample
   in the 3-D percentile space (PPL_pct, SE_pct, UMPIRE_pct).
   This ensures every baseline's score distribution overlaps.

The result is a 400-sample subset where baseline AUROC ≈ 0.5.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl_records

MODELS = ("llava", "qwen", "internvl")
METHODS = ("perplexity", "semantic_entropy", "umpire")
DATASETS = ("vilp", "hallusionbench", "mmvet")


def _avg_rank_pct(value: float, sorted_values: list[float]) -> float:
    lo = bisect.bisect_left(sorted_values, value) + 1
    hi = bisect.bisect_right(sorted_values, value)
    return ((lo + hi) / 2.0) / len(sorted_values)


def _load_uq(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    for row in load_jsonl_records(path)[1:]:
        sid = row.get("sample", {}).get("sample_id")
        if not isinstance(sid, str):
            continue
        uq = row.get("uq", {})
        scores = {}
        for m in METHODS:
            e = uq.get(m, {})
            s = e.get("score")
            if e.get("valid") is True and isinstance(s, (int, float)):
                scores[m] = float(s)
        if len(scores) == len(METHODS):
            result[sid] = scores
    return result


def _load_judge(path: Path) -> dict[str, dict]:
    result = {}
    for row in load_jsonl_records(path):
        if row.get("record_type") == "run":
            continue
        sid = row.get("sample", {}).get("sample_id")
        j = row.get("judge", {})
        if isinstance(sid, str) and j.get("valid") is True:
            result[sid] = {
                "hallucination": j.get("hallucination"),
                "group_id": row.get("sample", {}).get("group_id", sid),
                "dataset": row.get("sample", {}).get("dataset", "unknown"),
            }
    return result


def _load_has_image(path: Path) -> dict[str, bool]:
    """Load has_image from generation greedy records."""
    result = {}
    for row in load_jsonl_records(path):
        if row.get("record_type") != "sample":
            continue
        sid = row.get("sample", {}).get("sample_id")
        if isinstance(sid, str):
            result[sid] = row.get("sample", {}).get("has_image", True)
    return result


def _dist3(a: list[float], b: list[float]) -> float:
    """Euclidean distance in 3-D percentile space."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def extract_per_model(
    uq_dir: Path,
    judge_dir: Path,
    gen_dir: Path,
    model: str,
    target: int = 200,
) -> dict:
    # Load and join
    uq_all: dict[str, dict[str, float]] = {}
    judge_all: dict[str, dict] = {}
    has_image_all: dict[str, bool] = {}
    for ds in DATASETS:
        uq_all.update(_load_uq(uq_dir / model / f"{ds}.jsonl"))
        judge_all.update(_load_judge(judge_dir / model / f"{ds}.jsonl"))
        gen_path = gen_dir / model / "greedy" / f"{ds}.jsonl"
        if gen_path.exists():
            has_image_all.update(_load_has_image(gen_path))

    rows = []
    for sid in uq_all:
        if sid not in judge_all:
            continue
        j = judge_all[sid]
        if j["hallucination"] is None:
            continue
        if not has_image_all.get(sid, True):
            continue  # skip no-image samples
        rows.append({
            "sample_id": sid,
            "dataset": j["dataset"],
            "group_id": j["group_id"],
            **{f"{m}_score": uq_all[sid][m] for m in METHODS},
            "hallucination": j["hallucination"],
        })

    # Per-method percentile ranks
    for m in METHODS:
        key = f"{m}_score"
        sorted_vals = sorted(r[key] for r in rows)
        for r in rows:
            r[f"{m}_pct"] = _avg_rank_pct(r[key], sorted_vals)

    for r in rows:
        r["avg_pct"] = sum(r[f"{m}_pct"] for m in METHODS) / len(METHODS)

    hal = sorted([r for r in rows if r["hallucination"]], key=lambda r: r["avg_pct"])
    nonhal_pool = [r for r in rows if not r["hallucination"]]

    # Positive: 200 H=1 with lowest avg_pct
    positive = hal[:target]

    # Negative: greedy nearest-neighbour matching in 3-D percentile space
    used: set[str] = set()
    negative: list[dict] = []
    for pos in positive:
        pos_vec = [pos[f"{m}_pct"] for m in METHODS]
        best: dict | None = None
        best_dist = float("inf")
        for neg in nonhal_pool:
            if neg["sample_id"] in used:
                continue
            neg_vec = [neg[f"{m}_pct"] for m in METHODS]
            d = _dist3(pos_vec, neg_vec)
            if d < best_dist:
                best_dist = d
                best = neg
        if best is not None:
            used.add(best["sample_id"])
            negative.append(best)

    return {
        "model": model,
        "positive_ids": [r["sample_id"] for r in positive],
        "negative_ids": [r["sample_id"] for r in negative],
        "n_positive": len(positive),
        "n_negative": len(negative),
        "pos_avg_pct": [r["avg_pct"] for r in positive[:5]] + ["..."] + [r["avg_pct"] for r in positive[-5:]],
        "neg_avg_pct": [r["avg_pct"] for r in negative[:5]] + ["..."] + [r["avg_pct"] for r in negative[-5:]],
    }


def main():
    parser = argparse.ArgumentParser(description="Extract per-model LUH subsets with NN matching.")
    parser.add_argument("--uq-dir", required=True, type=Path)
    parser.add_argument("--judge-dir", required=True, type=Path)
    parser.add_argument("--gen-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=200)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Models to extract (default: all models).",
    )
    args = parser.parse_args()

    all_subsets = {}
    for model in args.models:
        result = extract_per_model(args.uq_dir, args.judge_dir, args.gen_dir, model, args.target_size)
        all_subsets[model] = result
        print(f"{model}: positive={result['n_positive']} negative={result['n_negative']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_subsets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for model, subset in all_subsets.items():
        sample_ids = subset["positive_ids"] + subset["negative_ids"]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"{model}: duplicate sample IDs in extracted subset")
        ids_path = args.output.parent / f"{model}_subset_ids.txt"
        ids_path.write_text("".join(f"{sid}\n" for sid in sample_ids), encoding="utf-8")
        print(f"{model}: ERA sample IDs -> {ids_path} ({len(sample_ids)})")

    # Quick baseline AUROC check
    print("\n=== Baseline AUROC check ===")
    for model in args.models:
        sub = all_subsets[model]
        all_ids = set(sub["positive_ids"] + sub["negative_ids"])
        pos_set = set(sub["positive_ids"])

        uq: dict[str, dict[str, float]] = {}
        judges: dict[str, bool] = {}
        for ds in DATASETS:
            for row in load_jsonl_records(args.uq_dir / model / f"{ds}.jsonl"):
                if row.get("record_type") != "sample":
                    continue
                sid = row.get("sample", {}).get("sample_id")
                if sid in all_ids:
                    scores = {}
                    for m in METHODS:
                        e = row.get("uq", {}).get(m, {})
                        if e.get("valid") and e.get("score") is not None:
                            scores[m] = float(e["score"])
                    if len(scores) == len(METHODS):
                        uq[sid] = scores
            for row in load_jsonl_records(args.judge_dir / model / f"{ds}.jsonl"):
                if row.get("record_type") == "run":
                    continue
                sid = row.get("sample", {}).get("sample_id")
                if sid in all_ids:
                    j = row.get("judge", {})
                    if j.get("valid"):
                        judges[sid] = bool(j.get("hallucination"))

        sids = sorted(s for s in all_ids if s in uq and s in judges)
        labels = [1 if judges[s] else 0 for s in sids]
        line = f"  {model} (n={len(sids)}): "
        for m in METHODS:
            scores = [uq[s][m] for s in sids]
            # AUROC
            pos_scores = [s for s, l in zip(scores, labels) if l == 1]
            neg_scores = [s for s, l in zip(scores, labels) if l == 0]
            if pos_scores and neg_scores:
                cnt = sum(1 for p in pos_scores for n in neg_scores if p > n)
                cnt += sum(0.5 for p in pos_scores for n in neg_scores if p == n)
                auc = cnt / (len(pos_scores) * len(neg_scores))
                line += f"{m[:3]}={auc:.3f} "
        print(line)

    print(f"\noutput={args.output}")


if __name__ == "__main__":
    main()
