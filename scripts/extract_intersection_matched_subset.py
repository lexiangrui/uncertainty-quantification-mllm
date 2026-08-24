#!/usr/bin/env python3
"""Extract a multi-method LUH intersection and matched H=0 controls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_per_model_subset import METHODS, MODELS, _avg_rank_pct, _dist3
from scripts.extract_quartile_luh_subsets import _load_pool


def extract_intersection_matched(rows: list[dict], alpha: float) -> dict:
    thresholds = {
        method: float(
            np.quantile(
                [row["scores"][method] for row in rows if not row["hallucination"]],
                alpha,
            )
        )
        for method in METHODS
    }
    sorted_values = {
        method: sorted(row["scores"][method] for row in rows) for method in METHODS
    }
    ranked = []
    for row in rows:
        item = dict(row)
        item["percentiles"] = {}
        for method in METHODS:
            item["percentiles"][method] = _avg_rank_pct(
                row["scores"][method], sorted_values[method]
            )
        item["avg_percentile"] = sum(item["percentiles"].values()) / len(METHODS)
        ranked.append(item)

    positives = sorted(
        (
            row
            for row in ranked
            if row["hallucination"]
            and all(row["scores"][method] <= thresholds[method] for method in METHODS)
        ),
        key=lambda row: (row["avg_percentile"], row["sample_id"]),
    )
    negative_pool = [row for row in ranked if not row["hallucination"]]
    used: set[str] = set()
    pairs = []
    for positive in positives:
        positive_vector = [positive["percentiles"][method] for method in METHODS]
        candidates = [row for row in negative_pool if row["sample_id"] not in used]
        if not candidates:
            raise ValueError("not enough non-hallucination rows for one-to-one matching")
        negative = min(
            candidates,
            key=lambda row: (
                _dist3(
                    positive_vector,
                    [row["percentiles"][method] for method in METHODS],
                ),
                row["sample_id"],
            ),
        )
        used.add(negative["sample_id"])
        distance = _dist3(
            positive_vector,
            [negative["percentiles"][method] for method in METHODS],
        )
        pairs.append(
            {
                "positive_id": positive["sample_id"],
                "negative_id": negative["sample_id"],
                "distance": distance,
            }
        )

    return {
        "positive_ids": [pair["positive_id"] for pair in pairs],
        "negative_ids": [pair["negative_id"] for pair in pairs],
        "n_positive": len(pairs),
        "n_negative": len(pairs),
        "alpha": alpha,
        "thresholds": thresholds,
        "definition": "H=1 and all three scores <= their Q_alpha(H0)",
        "negative_matching": "greedy one-to-one nearest H=0 in 3-D pooled percentile space",
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a three-method LUH intersection with matched negatives."
    )
    parser.add_argument("--uq-dir", required=True, type=Path)
    parser.add_argument("--judge-dir", required=True, type=Path)
    parser.add_argument("--gen-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be strictly between 0 and 1")

    rows = _load_pool(args.uq_dir, args.judge_dir, args.gen_dir, args.model)
    subset = extract_intersection_matched(rows, args.alpha)
    payload = {args.model: subset}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ids = subset["positive_ids"] + subset["negative_ids"]
    ids_path = args.output.parent / f"{args.model}_subset_ids.txt"
    ids_path.write_text("".join(f"{sample_id}\n" for sample_id in ids), encoding="utf-8")
    print(
        f"{args.model}: positive={subset['n_positive']} negative={subset['n_negative']} "
        f"total={len(ids)} -> {args.output}"
    )
    print(f"ERA sample IDs -> {ids_path}")


if __name__ == "__main__":
    main()
