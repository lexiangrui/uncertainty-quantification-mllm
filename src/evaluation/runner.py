from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import (
    auprc,
    auroc,
    bootstrap_summary,
    cluster_bootstrap_indices,
    prr,
)


SINGLE_CLASS_REASON = "target labels contain a single class"


def _load_records(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    run: dict[str, Any] | None = None
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if line_number == 1:
                if record.get("record_type") != "run" or not isinstance(record.get("run"), dict):
                    raise ValueError(f"missing run header at {path}:1")
                run = record["run"]
                continue
            if record.get("record_type") != "sample":
                raise ValueError(f"invalid sample record at {path}:{line_number}")
            sample_id = record.get("sample", {}).get("sample_id")
            if not isinstance(sample_id, str) or sample_id in records:
                raise ValueError(f"invalid or duplicate sample_id at {path}:{line_number}")
            records[sample_id] = record
    if run is None:
        raise ValueError(f"input is empty: {path}")
    return run, records


def _method_names(uq_run: dict[str, Any]) -> list[str]:
    configs = uq_run.get("uq_methods")
    if not isinstance(configs, list) or not configs:
        raise ValueError("uq run header lacks uq_methods")
    names: list[str] = []
    for config in configs:
        name = config.get("name") if isinstance(config, dict) else None
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("invalid uq method name in run header")
        names.append(name)
    return names


def _collect_rows(
    judge_records: dict[str, dict[str, Any]],
    uq_records: dict[str, dict[str, Any]],
    methods: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    invalid_judge = 0
    missing_uq = 0
    invalid_score_rows = 0
    invalid_score_by_method = {name: 0 for name in methods}
    for sample_id, record in judge_records.items():
        judge = record.get("judge")
        if not isinstance(judge, dict):
            raise ValueError(f"judge record lacks a judge object: {sample_id}")
        if judge.get("valid") is not True:
            invalid_judge += 1
            continue
        correct = judge.get("correct")
        hallucination = judge.get("hallucination")
        if type(correct) is not bool or type(hallucination) is not bool:
            raise ValueError(f"valid judge record lacks boolean labels: {sample_id}")
        group_id = record.get("sample", {}).get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"judge record lacks group_id: {sample_id}")
        uq_record = uq_records.get(sample_id)
        if uq_record is None:
            missing_uq += 1
            continue
        uq = uq_record.get("uq")
        if not isinstance(uq, dict):
            raise ValueError(f"uq record lacks a uq object: {sample_id}")
        scores: dict[str, float] = {}
        invalid_methods: list[str] = []
        for name in methods:
            value = uq.get(name)
            score = value.get("score") if isinstance(value, dict) and value.get("valid") is True else None
            if type(score) not in (int, float) or not math.isfinite(score):
                invalid_methods.append(name)
            else:
                scores[name] = float(score)
        if invalid_methods:
            invalid_score_rows += 1
            for name in invalid_methods:
                invalid_score_by_method[name] += 1
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "correct": int(correct),
                "hallucination": int(hallucination),
                "scores": scores,
            }
        )
    exclusions = {
        "invalid_judge": invalid_judge,
        "missing_uq_record": missing_uq,
        "invalid_uq_score": invalid_score_rows,
        "invalid_uq_score_by_method": invalid_score_by_method,
        "uq_without_judge_record": len(set(uq_records) - set(judge_records)),
    }
    return rows, exclusions


def _metric_entry(
    metric,
    scores: np.ndarray,
    target: np.ndarray,
    replicates: list[np.ndarray],
    *,
    confidence: float,
    **kwargs: Any,
) -> dict[str, Any]:
    value = metric(scores, target, **kwargs)
    if value is None:
        return {"value": None, "ci_low": None, "ci_high": None, "reason": SINGLE_CLASS_REASON}
    entry: dict[str, Any] = {"value": float(value)}
    entry.update(
        bootstrap_summary(
            lambda indices: metric(scores[indices], target[indices], **kwargs),
            replicates,
            confidence=confidence,
        )
    )
    return entry


def _rate_entry(
    values: np.ndarray, replicates: list[np.ndarray], *, confidence: float
) -> dict[str, Any]:
    entry: dict[str, Any] = {"value": float(values.mean())}
    entry.update(
        bootstrap_summary(
            lambda indices: float(values[indices].mean()),
            replicates,
            confidence=confidence,
        )
    )
    return entry


def run_metrics(
    *,
    uq_input: Path,
    judge_input: Path,
    output: Path,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    uq_run, uq_records = _load_records(uq_input)
    judge_run, judge_records = _load_records(judge_input)
    greedy_run = uq_run.get("greedy_run")
    if not isinstance(greedy_run, dict):
        raise ValueError("uq run header lacks greedy_run")
    if greedy_run != judge_run.get("greedy_run"):
        raise ValueError("greedy_run mismatch between UQ and judge inputs")
    methods = _method_names(uq_run)
    rows, exclusions = _collect_rows(judge_records, uq_records, methods)
    if not rows:
        raise ValueError("no samples left to evaluate after quality control")

    groups = [row["group_id"] for row in rows]
    correct = np.array([row["correct"] for row in rows], dtype=np.int64)
    hallucination = np.array([row["hallucination"] for row in rows], dtype=np.int64)
    error = 1 - correct
    scores = {
        name: np.array([row["scores"][name] for row in rows], dtype=np.float64)
        for name in methods
    }
    replicates = cluster_bootstrap_indices(
        groups, n_bootstrap=bootstrap_samples, seed=bootstrap_seed
    )

    labels_section = {
        "accuracy": _rate_entry(correct, replicates, confidence=confidence),
        "hallucination_rate": _rate_entry(hallucination, replicates, confidence=confidence),
        "joint_counts": {
            "correct_without_hallucination": int(((correct == 1) & (hallucination == 0)).sum()),
            "correct_with_hallucination": int(((correct == 1) & (hallucination == 1)).sum()),
            "wrong_without_hallucination": int(((correct == 0) & (hallucination == 0)).sum()),
            "wrong_with_hallucination": int(((correct == 0) & (hallucination == 1)).sum()),
        },
    }

    targets_section: dict[str, Any] = {}
    for target_name, target in (("error", error), ("hallucination", hallucination)):
        method_section: dict[str, Any] = {}
        for name in methods:
            method_scores = scores[name]
            method_section[name] = {
                "auroc": _metric_entry(
                    auroc, method_scores, target, replicates, confidence=confidence
                ),
                "auprc": _metric_entry(
                    auprc, method_scores, target, replicates, confidence=confidence
                ),
                "prr": _metric_entry(
                    prr, method_scores, target, replicates, confidence=confidence
                ),
            }
        positives = int(target.sum())
        targets_section[target_name] = {
            "positives": positives,
            "negatives": int(target.size - positives),
            "positive_rate": float(target.mean()),
            "methods": method_section,
        }

    report = {
        "uq_input": str(uq_input.resolve()),
        "judge_input": str(judge_input.resolve()),
        "dataset": greedy_run.get("dataset"),
        "model_family": greedy_run.get("model_family"),
        "model_id": greedy_run.get("model_id"),
        "uq_methods": methods,
        "config": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence": confidence,
            "cluster_field": "group_id",
            "positive_targets": {"error": "1 - correct", "hallucination": "hallucination"},
        },
        "counts": {
            "judge_records": len(judge_records),
            "uq_records": len(uq_records),
            "evaluated": len(rows),
            "clusters": len(set(groups)),
            "excluded": exclusions,
        },
        "labels": labels_section,
        "targets": targets_section,
        "uq_run": uq_run,
        "judge_run": judge_run,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
