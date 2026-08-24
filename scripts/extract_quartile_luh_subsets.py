#!/usr/bin/env python3
"""Extract nine method-specific LUH subsets using Experiment 1 quartiles.

For each model x UQ method, pool the three datasets using the same eligibility
rules as Experiment 1 C1 (valid judge, all three UQ scores valid, and an image).
The subset contains every hallucinated sample whose method score is at or below
the 25th percentile of that model/method's non-hallucination scores:

    score(H=1) <= Q_0.25(score(H=0))

No fixed target size or matched negative set is used. Ties at the threshold are
included, so subset sizes are intentionally data-dependent.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_per_model_subset import (  # noqa: E402
    DATASETS,
    METHODS,
    MODELS,
    _load_has_image,
    _load_judge,
    _load_uq,
)


def _load_pool(
    uq_dir: Path,
    judge_dir: Path,
    gen_dir: Path,
    model: str,
) -> list[dict]:
    uq_all: dict[str, dict[str, float]] = {}
    judge_all: dict[str, dict] = {}
    has_image_all: dict[str, bool] = {}
    for dataset in DATASETS:
        uq_all.update(_load_uq(uq_dir / model / f"{dataset}.jsonl"))
        judge_all.update(_load_judge(judge_dir / model / f"{dataset}.jsonl"))
        generation_path = gen_dir / model / "greedy" / f"{dataset}.jsonl"
        if generation_path.exists():
            has_image_all.update(_load_has_image(generation_path))

    rows: list[dict] = []
    for sample_id, scores in uq_all.items():
        judge = judge_all.get(sample_id)
        if judge is None or judge["hallucination"] is None:
            continue
        if not has_image_all.get(sample_id, True):
            continue
        if any(not math.isfinite(scores[method]) for method in METHODS):
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "dataset": judge["dataset"],
                "group_id": judge["group_id"],
                "hallucination": bool(judge["hallucination"]),
                "scores": scores,
            }
        )
    return rows


def select_quartile_luh(rows: list[dict], method: str, alpha: float) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    h0 = [row["scores"][method] for row in rows if not row["hallucination"]]
    h1 = [row for row in rows if row["hallucination"]]
    if not h0 or not h1:
        raise ValueError(f"cannot define LUH subset without both H=0 and H=1 rows: {method}")

    threshold = float(np.quantile(np.asarray(h0, dtype=float), alpha))
    selected = sorted(
        (row for row in h1 if row["scores"][method] <= threshold),
        key=lambda row: (row["scores"][method], row["dataset"], row["sample_id"]),
    )
    dataset_counts = Counter(row["dataset"] for row in selected)
    return {
        "method": method,
        "alpha": alpha,
        "threshold": threshold,
        "comparison": "score <= threshold",
        "n_evaluated": len(rows),
        "n_h0": len(h0),
        "n_h1": len(h1),
        "n_selected": len(selected),
        "luh_share": len(selected) / len(h1),
        "dataset_counts": {dataset: dataset_counts.get(dataset, 0) for dataset in DATASETS},
        "sample_ids": [row["sample_id"] for row in selected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract nine model x method LUH subsets using Q_alpha(H0)."
    )
    parser.add_argument("--uq-dir", required=True, type=Path)
    parser.add_argument("--judge-dir", required=True, type=Path)
    parser.add_argument("--gen-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be strictly between 0 and 1")

    manifest = {
        "schema_version": 1,
        "definition": "hallucination and score <= Q_alpha(non_hallucination scores)",
        "pool": "judge_valid + all_three_uq_valid + has_image; datasets pooled per model",
        "alpha": args.alpha,
        "subsets": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        rows = _load_pool(args.uq_dir, args.judge_dir, args.gen_dir, model)
        manifest["subsets"][model] = {}
        for method in args.methods:
            subset = select_quartile_luh(rows, method, args.alpha)
            manifest["subsets"][model][method] = subset
            ids_path = args.output_dir / f"{model}_{method}_subset_ids.txt"
            ids_path.write_text(
                "".join(f"{sample_id}\n" for sample_id in subset["sample_ids"]),
                encoding="utf-8",
            )
            print(
                f"{model}/{method}: selected={subset['n_selected']} "
                f"of H1={subset['n_h1']} threshold={subset['threshold']:.8g} "
                f"-> {ids_path}",
                flush=True,
            )

    manifest_path = args.output_dir / "quartile_luh_subsets.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output={manifest_path}")


if __name__ == "__main__":
    main()
