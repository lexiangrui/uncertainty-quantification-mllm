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
    out = {"accuracy": float("nan"), "metrics": {"uncertainty": {"auroc": float("nan"), "aupr": float("nan")}}}
    if not labels:
        return out
    out["accuracy"] = float(np.mean(labels))
    if len(set(labels)) < 2:
        return out
    y_true = np.array(labels)
    preds = -np.array(uncertainty, dtype=np.float64)
    finite = np.isfinite(preds)
    if int(finite.sum()) != len(preds):
        return out
    auroc = roc_auc_score(y_true, preds)
    aupr = average_precision_score(y_true, preds if auroc >= 0.5 else -preds)
    out["metrics"]["uncertainty"] = {
        "auroc": float(auroc),
        "aupr": float(aupr) if math.isfinite(aupr) else float("nan"),
    }
    return out
