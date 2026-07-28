"""AUROC / AUPR evaluation for VAUQ scores.

Convention: ``labels[i] == 1`` means the answer is *correct*. ``vauq`` and
``entropy`` are negated before scoring (lower VAUQ => more likely correct =>
should rank correct answers higher); ``is_score`` is not. AUPR flips the sign
when AUROC < 0.5.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SCORE_NAMES = ("vauq", "entropy", "is_score")


def compute_metrics(labels: list[int], vauq: list[float], entropy: list[float], is_score: list[float]) -> dict:
    """Return ``{name: {auroc, aupr}}`` plus accuracy. NaN if labels are single-class."""
    if not labels:
        raise ValueError("evaluation requires labeled records")
    if len(set(labels)) < 2:
        raise ValueError("AUROC requires both correct and incorrect answers")

    y_true = np.array(labels)
    out: dict = {"accuracy": float(np.mean(labels)), "metrics": {}}
    raw_map = {"vauq": vauq, "entropy": entropy, "is_score": is_score}
    for name in SCORE_NAMES:
        preds = np.array(raw_map[name], dtype=np.float64)
        if name in ("vauq", "entropy"):
            preds = -preds  # lower VAUQ/entropy => more likely correct
        finite = np.isfinite(preds)
        if int(finite.sum()) != len(preds):
            raise ValueError(f"{name} scores must all be finite")
        auroc = roc_auc_score(y_true, preds)
        aupr = average_precision_score(y_true, preds if auroc >= 0.5 else -preds)
        if not math.isfinite(aupr):
            raise ValueError(f"{name} AUPR is not finite")
        out["metrics"][name] = {"auroc": float(auroc), "aupr": float(aupr)}
    return out
