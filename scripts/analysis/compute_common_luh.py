#!/usr/bin/env python3
"""Compute common and core low-uncertainty hallucinations across MLLMs."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl_records


MODELS = ("llava", "qwen", "internvl")
DATASETS = ("vilp", "hallusionbench", "mmvet")
METHODS = ("perplexity", "semantic_entropy", "umpire")


def _scores(path: Path) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in load_jsonl_records(path)[1:]:
        sample_id = row.get("sample", {}).get("sample_id")
        uq = row.get("uq")
        if not isinstance(sample_id, str) or not isinstance(uq, dict):
            continue
        values: dict[str, float] = {}
        for method in METHODS:
            entry = uq.get(method)
            score = entry.get("score") if isinstance(entry, dict) else None
            if (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(score)
                and entry.get("valid") is True
            ):
                values[method] = float(score)
        if len(values) == len(METHODS):
            output[sample_id] = values
    return output


def _hallucinations(path: Path) -> dict[str, bool]:
    output: dict[str, bool] = {}
    for row in load_jsonl_records(path)[1:]:
        sample_id = row.get("sample", {}).get("sample_id")
        judge = row.get("judge")
        if (
            isinstance(sample_id, str)
            and isinstance(judge, dict)
            and judge.get("valid") is True
            and type(judge.get("hallucination")) is bool
        ):
            output[sample_id] = judge["hallucination"]
    return output


def average_rank_percentiles(values: dict[str, float]) -> dict[str, float]:
    """Return average-rank percentiles in (0, 1], preserving score ties."""
    ordered = sorted(values.values())
    size = len(ordered)
    result: dict[str, float] = {}
    for sample_id, value in values.items():
        first = bisect.bisect_left(ordered, value) + 1
        last = bisect.bisect_right(ordered, value)
        result[sample_id] = ((first + last) / 2.0) / size
    return result


def compute_common_luh(
    *, uq_root: Path, judge_root: Path, low_quantile: float
) -> dict:
    cells: dict[tuple[str, str], dict[str, dict]] = {}
    universe: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}

    for dataset in DATASETS:
        for model in MODELS:
            score_rows = _scores(uq_root / model / f"{dataset}.jsonl")
            hallucination = _hallucinations(judge_root / model / f"{dataset}.jsonl")
            percentiles = {
                method: average_rank_percentiles(
                    {
                        sample_id: values[method]
                        for sample_id, values in score_rows.items()
                    }
                )
                for method in METHODS
            }
            model_rows: dict[str, dict] = {}
            for sample_id, scores in score_rows.items():
                ranks = {
                    method: percentiles[method][sample_id] for method in METHODS
                }
                low = {method: ranks[method] <= low_quantile for method in METHODS}
                low_count = sum(low.values())
                hallucinated = hallucination.get(sample_id)
                model_rows[sample_id] = {
                    "judge_available": hallucinated is not None,
                    "hallucination": hallucinated,
                    "scores": scores,
                    "percentiles": ranks,
                    "low_uq": low,
                    "low_method_count": low_count,
                    "model_luh": hallucinated is True and low_count >= 2,
                }
            cells[(dataset, model)] = model_rows
            universe[dataset].update(model_rows)

    common_samples: list[dict] = []
    core_samples: list[dict] = []
    dataset_counts: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        common_count = core_count = 0
        for sample_id in sorted(universe[dataset]):
            models = {
                model: cells[(dataset, model)].get(
                    sample_id,
                    {
                        "judge_available": False,
                        "hallucination": None,
                        "scores": {},
                        "percentiles": {},
                        "low_uq": {},
                        "low_method_count": 0,
                        "model_luh": False,
                    },
                )
                for model in MODELS
            }
            model_luh_count = sum(item["model_luh"] for item in models.values())
            record = {
                "dataset": dataset,
                "sample_id": sample_id,
                "model_luh_count": model_luh_count,
                "models": models,
            }
            if model_luh_count >= 2:
                common_samples.append(record)
                common_count += 1
            if model_luh_count == 3:
                core_samples.append(record)
                core_count += 1
        dataset_counts[dataset] = {
            "common_luh": common_count,
            "core_luh": core_count,
        }

    return {
        "definition": {
            "percentile_scope": "dataset_x_model_x_method",
            "percentile_ties": "average_rank",
            "low_uq_threshold": low_quantile,
            "model_luh": "hallucination and at least 2 of 3 low-UQ methods",
            "common_luh": "model_luh on at least 2 of 3 models",
            "core_luh": "model_luh on all 3 models",
        },
        "counts": {
            "common_luh": len(common_samples),
            "core_luh": len(core_samples),
            "by_dataset": dataset_counts,
        },
        "common_samples": common_samples,
        "core_sample_ids": [
            {"dataset": item["dataset"], "sample_id": item["sample_id"]}
            for item in core_samples
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uq-root", type=Path, default=PROJECT_ROOT / "results" / "uq")
    parser.add_argument(
        "--judge-root", type=Path, default=PROJECT_ROOT / "results" / "judging"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "analysis" / "luh" / "common_luh.json",
    )
    parser.add_argument("--low-quantile", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.0 < args.low_quantile < 1.0:
        parser.error("--low-quantile must be between 0 and 1")

    result = compute_common_luh(
        uq_root=args.uq_root,
        judge_root=args.judge_root,
        low_quantile=args.low_quantile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = result["counts"]
    print(
        f"common_luh={counts['common_luh']} core_luh={counts['core_luh']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
