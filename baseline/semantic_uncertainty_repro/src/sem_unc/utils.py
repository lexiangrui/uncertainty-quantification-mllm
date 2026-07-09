"""Utility functions: prompt building, serialization, logging."""

from __future__ import annotations

import json
import logging
import os
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np


def setup_logger(level: int = logging.INFO) -> None:
    """Configure root logger with timestamps."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, returning *default* if unset."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write a list of dicts as JSONL (one object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict) -> None:
    """Write a dict as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_answer(s: str) -> str:
    """Lower-case, remove articles, punctuation, and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def squad_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between *prediction* and *ground_truth*."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_bootstrap(
    data: list[float],
    func: callable = np.mean,
    n_resamples: int = 1000,
    confidence_level: float = 0.9,
    seed: int = 41,
) -> dict[str, float]:
    """Return {mean, std_err, low, high} via bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(data)
    estimates = []
    for _ in range(n_resamples):
        sample = rng.choice(data, size=n, replace=True)
        estimates.append(func(sample))
    estimates = np.array(estimates)
    alpha = (1 - confidence_level) / 2
    return {
        "mean": np.mean(estimates),
        "std_err": np.std(estimates, ddof=1),
        "low": np.quantile(estimates, alpha),
        "high": np.quantile(estimates, 1 - alpha),
    }


def get_metric_fn(metric_name: str) -> callable:
    """Return a scoring function given a metric name.

    Currently supported: ``"squad"`` (F1 ≥ 50), ``"exact_match"``.
    """
    if metric_name == "squad":

        def _metric(prediction: str, ground_truth: str) -> float:
            return 1.0 if squad_f1_score(prediction, ground_truth) >= 0.5 else 0.0

        return _metric

    if metric_name == "exact_match":

        def _metric(prediction: str, ground_truth: str) -> float:
            return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

        return _metric

    raise ValueError(f"Unknown metric: {metric_name}")
