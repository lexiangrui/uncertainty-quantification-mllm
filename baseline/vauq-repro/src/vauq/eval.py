"""AUROC / AUPR evaluation for VAUQ scores.

Convention: ``labels[i] == 1`` means the answer is *correct*. ``vauq`` and
``entropy`` are negated before scoring (lower VAUQ => more likely correct =>
should rank correct answers higher); ``is_score`` is not. AUPR flips the sign
when AUROC < 0.5.
"""

from __future__ import annotations

import hashlib
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


def compute_conformal_metrics(
    records: list[dict],
    calibration_fraction: float = 0.3,
    seed: int = 42,
) -> dict:
    """Evaluate uncertainty with a split-conformal visual evidence score.

    The nonconformity score is two-dimensional:

    - lower original-image entropy is more conforming;
    - larger intervention effect (``entropy_masked - entropy``) is more conforming.

    Calibration uses only correct examples, so the conformal p-value estimates
    how much the current sample resembles previously correct, visually grounded
    answers. AUROC is reported with the error label as positive, using
    ``1 - p_value`` as uncertainty.
    """

    labeled = [record for record in records if record.get("correct") is not None]
    if len(labeled) < 2:
        return _empty_conformal_metrics(len(labeled), 0, 0)

    calibration, test = _deterministic_split(labeled, calibration_fraction, seed)
    calibration_correct = [record for record in calibration if bool(record.get("correct"))]
    if not calibration_correct or len(test) < 2:
        return _empty_conformal_metrics(len(labeled), len(calibration_correct), len(test))

    cal_scores = np.array([_visual_evidence_vector(record) for record in calibration_correct], dtype=np.float64)
    test_scores = np.array([_visual_evidence_vector(record) for record in test], dtype=np.float64)
    if not np.isfinite(cal_scores).all() or not np.isfinite(test_scores).all():
        return _empty_conformal_metrics(len(labeled), len(calibration_correct), len(test))

    cal_mean = cal_scores.mean(axis=0)
    cal_std = cal_scores.std(axis=0)
    cal_std[cal_std < 1e-8] = 1.0

    cal_nonconformity = _visual_nonconformity(cal_scores, cal_mean, cal_std)
    test_nonconformity = _visual_nonconformity(test_scores, cal_mean, cal_std)
    p_values = np.array(
        [
            (1.0 + float(np.sum(cal_nonconformity >= score))) / (len(cal_nonconformity) + 1.0)
            for score in test_nonconformity
        ],
        dtype=np.float64,
    )
    uncertainty = 1.0 - p_values
    error_labels = np.array([0 if bool(record.get("correct")) else 1 for record in test], dtype=np.int64)

    out = {
        "n_labeled": len(labeled),
        "n_calibration": len(calibration),
        "n_calibration_correct": len(calibration_correct),
        "n_test": len(test),
        "calibration_fraction": float(calibration_fraction),
        "seed": int(seed),
        "score": "split_conformal_visual_evidence",
        "uncertainty": "1 - p_value",
        "auroc": float("nan"),
        "aupr": float("nan"),
        "mean_p_value": float(np.mean(p_values)),
        "mean_uncertainty": float(np.mean(uncertainty)),
    }
    if len(set(error_labels.tolist())) < 2:
        return out
    finite = np.isfinite(uncertainty)
    if int(finite.sum()) != len(uncertainty):
        return out
    auroc = roc_auc_score(error_labels, uncertainty)
    aupr = average_precision_score(error_labels, uncertainty if auroc >= 0.5 else -uncertainty)
    out["auroc"] = float(auroc)
    out["aupr"] = float(aupr) if math.isfinite(aupr) else float("nan")
    return out


def _empty_conformal_metrics(n_labeled: int, n_calibration_correct: int, n_test: int) -> dict:
    return {
        "n_labeled": n_labeled,
        "n_calibration": 0,
        "n_calibration_correct": n_calibration_correct,
        "n_test": n_test,
        "calibration_fraction": float("nan"),
        "seed": None,
        "score": "split_conformal_visual_evidence",
        "uncertainty": "1 - p_value",
        "auroc": float("nan"),
        "aupr": float("nan"),
        "mean_p_value": float("nan"),
        "mean_uncertainty": float("nan"),
    }


def _deterministic_split(records: list[dict], calibration_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    calibration_fraction = min(max(float(calibration_fraction), 0.05), 0.9)
    keyed = sorted(((_stable_unit_interval(record, seed), record) for record in records), key=lambda item: item[0])
    n_calibration = int(round(len(keyed) * calibration_fraction))
    n_calibration = min(max(1, n_calibration), len(keyed) - 1)
    calibration = [record for _, record in keyed[:n_calibration]]
    test = [record for _, record in keyed[n_calibration:]]
    return calibration, test


def _stable_unit_interval(record: dict, seed: int) -> float:
    key = f"{seed}:{record.get('id', '')}:{record.get('question', '')}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64 - 1)


def _visual_evidence_vector(record: dict) -> tuple[float, float]:
    scores = record.get("scores") or {}
    entropy = float(scores.get("entropy", float("nan")))
    is_score = float(scores.get("is_score", float("nan")))
    return -entropy, is_score


def _visual_nonconformity(scores: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (scores - mean) / std
    evidence = z[:, 0] + z[:, 1]
    return -evidence
