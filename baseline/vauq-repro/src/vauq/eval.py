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
    out: dict = {"accuracy": float("nan"), "metrics": {}}
    if not labels:
        return out
    out["accuracy"] = float(np.mean(labels))
    if len(set(labels)) < 2:
        for name in SCORE_NAMES:
            out["metrics"][name] = {"auroc": float("nan"), "aupr": float("nan")}
        return out

    y_true = np.array(labels)
    raw_map = {"vauq": vauq, "entropy": entropy, "is_score": is_score}
    for name in SCORE_NAMES:
        preds = np.array(raw_map[name], dtype=np.float64)
        if name in ("vauq", "entropy"):
            preds = -preds  # lower VAUQ/entropy => more likely correct
        finite = np.isfinite(preds)
        if int(finite.sum()) != len(preds) or len(set(labels[i] for i in range(len(labels)) if finite[i])) < 2:
            out["metrics"][name] = {"auroc": float("nan"), "aupr": float("nan")}
            continue
        auroc = roc_auc_score(y_true, preds)
        aupr = average_precision_score(y_true, preds if auroc >= 0.5 else -preds)
        out["metrics"][name] = {"auroc": float(auroc), "aupr": float(aupr) if math.isfinite(aupr) else float("nan")}
    return out
