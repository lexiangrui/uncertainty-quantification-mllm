"""Evaluation metrics for uncertainty scores."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(labels: list[int], uncertainty: list[float]) -> dict:
    """Compute accuracy, AUROC and AUPR.

    Labels use ``1`` for correct answers. Higher uncertainty should indicate
    incorrect answers, so scores are negated before ranking correctness.
    """
    if not labels:
        raise ValueError("evaluation requires labeled records")
    if len(set(labels)) < 2:
        raise ValueError("AUROC requires both correct and incorrect answers")
    y_true = np.array(labels)
    preds = -np.array(uncertainty, dtype=np.float64)
    finite = np.isfinite(preds)
    if int(finite.sum()) != len(preds):
        raise ValueError("uncertainty scores must all be finite")
    auroc = roc_auc_score(y_true, preds)
    aupr = average_precision_score(y_true, preds if auroc >= 0.5 else -preds)
    if not math.isfinite(aupr):
        raise ValueError("AUPR is not finite")
    return {
        "accuracy": float(np.mean(labels)),
        "metrics": {"uncertainty": {"auroc": float(auroc), "aupr": float(aupr)}},
    }
