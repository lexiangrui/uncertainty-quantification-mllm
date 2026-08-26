#!/usr/bin/env python3
"""Summarize dual-judge agreement and human-adjudication workload."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.human_alignment.workflow import (  # noqa: E402
    DATASETS,
    MODELS,
    _load_judge_file,
)

FIELDS = ("correct", "hallucination")


def cohens_kappa(left: Iterable[bool], right: Iterable[bool]) -> float:
    """Return Cohen's kappa for two binary label sequences."""
    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values) or not left_values:
        raise ValueError("kappa requires two non-empty label sequences of equal length")
    n = len(left_values)
    observed = sum(a == b for a, b in zip(left_values, right_values, strict=True)) / n
    left_positive = sum(left_values) / n
    right_positive = sum(right_values) / n
    expected = (
        left_positive * right_positive
        + (1.0 - left_positive) * (1.0 - right_positive)
    )
    if math.isclose(expected, 1.0):
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def summarize_labels(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    left = [bool(row["gpt"][field]) for row in rows]
    right = [bool(row["gemini"][field]) for row in rows]
    agreements = sum(a == b for a, b in zip(left, right, strict=True))
    return {
        "n": len(rows),
        "agreements": agreements,
        "disagreements": len(rows) - agreements,
        "agreement_rate": agreements / len(rows),
        "cohens_kappa": cohens_kappa(left, right),
        "gpt_positive_rate": sum(left) / len(left),
        "gemini_positive_rate": sum(right) / len(right),
    }


def _load_pairs(gpt_dir: Path, gemini_dir: Path) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for model in MODELS:
        for dataset in DATASETS:
            _, gpt_rows = _load_judge_file(gpt_dir / model / f"{dataset}.jsonl")
            _, gemini_rows = _load_judge_file(gemini_dir / model / f"{dataset}.jsonl")
            if set(gpt_rows) != set(gemini_rows):
                raise ValueError(f"sample set mismatch: {model}/{dataset}")
            for sample_id in sorted(gpt_rows):
                pairs.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "gpt": gpt_rows[sample_id]["judge"],
                        "gemini": gemini_rows[sample_id]["judge"],
                    }
                )
    return pairs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_adjudication(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count disputed fields and unique adjudicated samples by model."""
    rows: list[dict[str, Any]] = []
    for model in (*MODELS, "overall"):
        selected = samples if model == "overall" else [row for row in samples if row["model"] == model]
        correctness = sum(bool(row["disagreements"]["correct"]) for row in selected)
        hallucination = sum(bool(row["disagreements"]["hallucination"]) for row in selected)
        both = sum(
            bool(row["disagreements"]["correct"])
            and bool(row["disagreements"]["hallucination"])
            for row in selected
        )
        rows.append(
            {
                "model": model,
                "correctness_fields": correctness,
                "hallucination_fields": hallucination,
                "both_fields": both,
                "unique_samples": len(selected),
            }
        )
    return rows


def analyze(
    *, gpt_dir: Path, gemini_dir: Path, workspace: Path, output_dir: Path
) -> dict[str, Any]:
    pairs = _load_pairs(gpt_dir, gemini_dir)
    queue = json.loads((workspace / "samples.json").read_text(encoding="utf-8"))
    annotations = json.loads(
        (workspace / "annotations.json").read_text(encoding="utf-8")
    ).get("annotations", {})
    counts = queue.get("counts", {})
    if len(annotations) != counts.get("unique"):
        raise ValueError("human-adjudication annotations are incomplete")

    rows: list[dict[str, Any]] = []
    for model in (*MODELS, "overall"):
        selected = pairs if model == "overall" else [row for row in pairs if row["model"] == model]
        for field in FIELDS:
            rows.append({"model": model, "field": field, **summarize_labels(selected, field)})

    adjudication_rows = summarize_adjudication(queue.get("samples", []))
    overall_adjudication = next(row for row in adjudication_rows if row["model"] == "overall")
    expected = {
        "correctness_fields": int(counts.get("correct", 0)),
        "hallucination_fields": int(counts.get("hallucination", 0)),
        "both_fields": int(counts.get("overlap", 0)),
        "unique_samples": int(counts.get("unique", 0)),
    }
    if any(overall_adjudication[key] != value for key, value in expected.items()):
        raise ValueError("human-adjudication queue counts are internally inconsistent")

    summary = {
        "n_samples": len(pairs),
        "agreement": rows,
        "human_adjudication": adjudication_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "dual_judge_agreement.csv", rows)
    _write_csv(output_dir / "human_adjudication_counts.csv", adjudication_rows)
    (output_dir / "judge_reliability.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpt-dir", type=Path, default=PROJECT_ROOT / "results/judging_gpt_5_6_terra"
    )
    parser.add_argument(
        "--gemini-dir",
        type=Path,
        default=PROJECT_ROOT / "results/judging_gemini_3_7_flash",
    )
    parser.add_argument(
        "--workspace", type=Path, default=PROJECT_ROOT / "results/human_alignment"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/analysis/judge_reliability"
    )
    args = parser.parse_args()
    summary = analyze(
        gpt_dir=args.gpt_dir,
        gemini_dir=args.gemini_dir,
        workspace=args.workspace,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
