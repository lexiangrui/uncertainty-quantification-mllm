#!/usr/bin/env python3
"""Extract Low-Uncertainty Hallucination (LUH) subset and matched negatives.

Strategy:
1. Per model: merge 3 datasets, compute UQ percentile ranks per method.
2. Average percentile across 3 methods → avg_pct.
3. LUH positive: H=1 samples with lowest avg_pct. Try common intersection
   across all 3 models; fall back to per-model union if too few.
4. Negative: H=0 samples with avg_pct matched to LUH range.
"""
from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl_records

MODELS = ("llava", "qwen", "internvl")
METHODS = ("perplexity", "semantic_entropy", "umpire")
DATASETS = ("vilp", "hallusionbench", "mmvet")


def _avg_rank_pct(value: float, sorted_values: list[float]) -> float:
    """Average-rank percentile in (0, 1], handling ties."""
    lo = bisect.bisect_left(sorted_values, value) + 1
    hi = bisect.bisect_right(sorted_values, value)
    return ((lo + hi) / 2.0) / len(sorted_values)


def _load_uq_scores(path: Path) -> dict[str, dict[str, float]]:
    result = {}
    for row in load_jsonl_records(path)[1:]:
        sid = row.get("sample", {}).get("sample_id")
        uq = row.get("uq", {})
        if not isinstance(sid, str):
            continue
        scores = {}
        for m in METHODS:
            entry = uq.get(m, {})
            s = entry.get("score")
            if entry.get("valid") is True and isinstance(s, (int, float)):
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
        judge = row.get("judge", {})
        if isinstance(sid, str) and judge.get("valid") is True:
            result[sid] = {
                "hallucination": judge.get("hallucination"),
                "correct": judge.get("correct"),
                "group_id": row.get("sample", {}).get("group_id", sid),
                "dataset": row.get("sample", {}).get("dataset", "unknown"),
            }
    return result


def build_model_table(uq_root: Path, judge_root: Path, model: str) -> list[dict]:
    """Join UQ + judge for one model, compute percentile ranks."""
    uq_all: dict[str, dict[str, float]] = {}
    judge_all: dict[str, dict] = {}
    dataset_map: dict[str, str] = {}

    for ds in DATASETS:
        uq_path = uq_root / model / f"{ds}.jsonl"
        judge_path = judge_root / model / f"{ds}.jsonl"
        if not uq_path.exists() or not judge_path.exists():
            continue
        uq_all.update(_load_uq_scores(uq_path))
        j = _load_judge(judge_path)
        judge_all.update(j)
        for sid in j:
            dataset_map[sid] = ds

    rows = []
    for sid in uq_all:
        if sid not in judge_all:
            continue
        j = judge_all[sid]
        if j["hallucination"] is None:
            continue
        rows.append({
            "sample_id": sid,
            "model": model,
            "dataset": dataset_map.get(sid, "unknown"),
            "group_id": j["group_id"],
            **{f"{m}_score": uq_all[sid][m] for m in METHODS},
            "hallucination": j["hallucination"],
            "correct": j["correct"],
        })

    # Percentile ranks per method
    for m in METHODS:
        key = f"{m}_score"
        sorted_vals = sorted(r[key] for r in rows)
        for r in rows:
            r[f"{m}_pct"] = _avg_rank_pct(r[key], sorted_vals)

    for r in rows:
        r["avg_pct"] = statistics.mean(r[f"{m}_pct"] for m in METHODS)

    return rows


def extract_subset(
    model_tables: dict[str, list[dict]],
    target_size: int = 200,
) -> dict:
    # --- Per-model LUH candidates ---
    luh_per_model: dict[str, list[str]] = {}
    neg_pool_per_model: dict[str, list[dict]] = {}

    for model in MODELS:
        rows = model_tables.get(model, [])
        hal = sorted([r for r in rows if r["hallucination"]], key=lambda r: r["avg_pct"])
        nonhal = [r for r in rows if not r["hallucination"]]
        luh_per_model[model] = [r["sample_id"] for r in hal[:target_size]]

        if hal:
            max_pct = hal[min(target_size, len(hal)) - 1]["avg_pct"]
            neg_low = [r for r in nonhal if r["avg_pct"] <= max_pct]
            if len(neg_low) < target_size:
                neg_low = sorted(nonhal, key=lambda r: r["avg_pct"])[:target_size]
            neg_pool_per_model[model] = sorted(neg_low, key=lambda r: r["avg_pct"])

    # --- Common LUH intersection ---
    common_luh = set(luh_per_model[MODELS[0]])
    for m in MODELS[1:]:
        common_luh &= set(luh_per_model.get(m, []))

    if len(common_luh) >= target_size:
        strategy = "common_intersection"
        # Rank by cross-model avg pct
        table_lookup = {m: {r["sample_id"]: r for r in model_tables[m]} for m in MODELS}
        common_sorted = sorted(common_luh, key=lambda sid: statistics.mean(
            table_lookup[m][sid]["avg_pct"] for m in MODELS if sid in table_lookup.get(m, {})
        ))
        positive_ids = common_sorted[:target_size]
    else:
        strategy = "per_model_union"
        seen = set()
        positive_ids = []
        for m in MODELS:
            for sid in luh_per_model.get(m, []):
                if sid not in seen:
                    seen.add(sid)
                    positive_ids.append(sid)
                    if len(positive_ids) >= target_size:
                        break
            if len(positive_ids) >= target_size:
                break
        positive_ids = positive_ids[:target_size]

    # --- Negative selection ---
    pos_set = set(positive_ids)
    primary_model = max(neg_pool_per_model, key=lambda m: len(neg_pool_per_model.get(m, [])))
    neg_pool = [r for r in neg_pool_per_model.get(primary_model, []) if r["sample_id"] not in pos_set]
    negative_ids = [r["sample_id"] for r in neg_pool[:target_size]]

    return {
        "strategy": strategy,
        "positive_ids": positive_ids,
        "negative_ids": negative_ids,
        "positive_model": "common" if strategy == "common_intersection" else primary_model,
        "negative_model": primary_model,
        "common_luh_count": len(common_luh),
        "total_luh_per_model": {m: len(v) for m, v in luh_per_model.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract LUH subset and matched negatives.")
    parser.add_argument("--uq-dir", required=True, type=Path)
    parser.add_argument("--judge-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=200)
    args = parser.parse_args()

    model_tables = {}
    for model in MODELS:
        model_tables[model] = build_model_table(args.uq_dir, args.judge_dir, model)
        print(f"{model}: {len(model_tables[model])} joined samples")

    result = extract_subset(model_tables, args.target_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"strategy={result['strategy']} positive={len(result['positive_ids'])} "
          f"negative={len(result['negative_ids'])} common_luh={result['common_luh_count']}")


if __name__ == "__main__":
    main()
