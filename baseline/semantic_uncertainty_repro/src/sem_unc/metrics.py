"""Evaluation metrics for uncertainty quantification.

AUROC, area under thresholded accuracy (AURAC), and accuracy at
quantile — the three primary metrics used in the semantic uncertainty
paper.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn import metrics as skm

from sem_unc.utils import compute_bootstrap

logger = logging.getLogger(__name__)


def auroc(y_true: list[float], y_score: list[float]) -> float:
    """Area under the ROC curve.

    ``y_true``: 0 = correct, 1 = incorrect (higher score = more uncertain).
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true_arr)) < 2:
        logger.warning("AUROC: only one class present, returning 0.5.")
        return 0.5
    fpr, tpr, _ = skm.roc_curve(y_true_arr, y_score_arr)  # noqa: F841
    return float(skm.auc(fpr, tpr))


def accuracy_at_quantile(
    accuracies: list[float],
    uncertainties: list[float],
    quantile: float,
) -> float:
    """Accuracy among the bottom *quantile* of uncertainty scores.

    Lower uncertainty → we trust the model more → select those with
    uncertainty ≤ the *quantile*-th percentile.
    """
    cutoff = np.quantile(uncertainties, quantile)
    mask = np.asarray(uncertainties) <= cutoff
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.asarray(accuracies)[mask]))


def area_under_thresholded_accuracy(
    accuracies: list[float],
    uncertainties: list[float],
    n_quantiles: int = 20,
) -> float:
    """Area under the accuracy-vs-quantile curve (AURAC)."""
    quantiles = np.linspace(0.1, 1, n_quantiles)
    vals = [
        accuracy_at_quantile(accuracies, uncertainties, q) for q in quantiles
    ]
    dx = quantiles[1] - quantiles[0]
    return float(np.sum(vals) * dx)


def compute_metrics(
    labels: list[bool],
    semantic_entropy: list[float],
    regular_entropy: list[float] | None = None,
    cluster_entropy: list[float] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 41,
) -> dict:
    """Compute AUROC, AURAC, and accuracy-at-quantile for each score type.

    Parameters
    ----------
    labels:
        Per-sample correctness (``True`` = correct, ``False`` = incorrect).
    semantic_entropy:
        Per-sample semantic entropy values.
    regular_entropy:
        Per-sample regular (naive) entropy values.
    cluster_entropy:
        Per-sample cluster assignment entropy values.
    n_bootstrap:
        Number of bootstrap resamples.
    seed:
        Random seed for bootstrap.

    Returns
    -------
    summary : dict
        Nested dict with keys for each score type and aggregate metrics.
    """
    is_false = [0.0 if lab else 1.0 for lab in labels]
    accuracy = np.mean(labels)

    score_map = {"semantic_entropy": semantic_entropy}
    if regular_entropy is not None:
        score_map["regular_entropy"] = regular_entropy
    if cluster_entropy is not None:
        score_map["cluster_assignment_entropy"] = cluster_entropy

    result: dict = {
        "accuracy": accuracy,
        "metrics": {},
    }

    for name, scores in score_map.items():
        if len(scores) != len(is_false):
            logger.warning(
                "Mismatched lengths for '%s': %d scores vs %d labels.",
                name,
                len(scores),
                len(is_false),
            )
            continue

        auc = auroc(is_false, scores)
        aurac_val = area_under_thresholded_accuracy(labels, scores)

        # Accuracy at select quantiles.
        acc_at_q = {}
        for q in [0.8, 0.9, 0.95, 1.0]:
            acc_at_q[f"accuracy_at_{q}"] = accuracy_at_quantile(
                labels, scores, q
            )

        # Bootstrap confidence intervals.
        rng = np.random.default_rng(seed)
        auc_bs = _bootstrap_auroc(is_false, scores, n_bootstrap, rng)

        result["metrics"][name] = {
            "auroc": auc,
            "auroc_bootstrap": auc_bs,
            "aurac": aurac_val,
            **acc_at_q,
        }

    return result


def _bootstrap_auroc(
    y_true: list[float],
    y_score: list[float],
    n_resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Bootstrap AUROC with confidence intervals."""
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)
    n = len(yt)
    estimates = []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        est = auroc(yt[idx].tolist(), ys[idx].tolist())
        estimates.append(est)
    estimates = np.array(estimates)
    return {
        "mean": float(np.mean(estimates)),
        "std_err": float(np.std(estimates, ddof=1)),
        "low": float(np.quantile(estimates, 0.05)),
        "high": float(np.quantile(estimates, 0.95)),
    }
