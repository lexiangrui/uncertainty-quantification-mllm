from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def _validated(scores: Sequence[float], labels: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels)
    if score_array.ndim != 1 or score_array.shape != label_array.shape:
        raise ValueError("scores and labels must be one-dimensional and equally sized")
    if score_array.size == 0:
        raise ValueError("scores and labels must be non-empty")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    return score_array, label_array.astype(np.int64)


def _midranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    ends = np.r_[starts[1:], scores.size]
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.repeat((starts + 1 + ends) / 2.0, ends - starts)
    return ranks


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """P(s+ > s-) + 0.5 * P(s+ = s-); None when only one class is present."""
    score_array, label_array = _validated(scores, labels)
    positives = int(label_array.sum())
    negatives = label_array.size - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _midranks(score_array)
    rank_sum = float(ranks[label_array == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auprc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Average precision with tied scores collapsed into one threshold step."""
    score_array, label_array = _validated(scores, labels)
    positives = int(label_array.sum())
    if positives == 0 or positives == label_array.size:
        return None
    order = np.argsort(-score_array, kind="stable")
    sorted_labels = label_array[order]
    sorted_scores = score_array[order]
    group_last = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    true_positives = np.cumsum(sorted_labels)[group_last]
    predicted_positives = np.flatnonzero(group_last) + 1
    precision = true_positives / predicted_positives
    recall = true_positives / positives
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def _rejection_area(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under retained precision vs rejection rate, rejecting by score descending.

    Tied scores are handled by taking the exact expectation over all
    orderings within each tie group.
    """
    count = labels.size
    positives = float(labels.sum())
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    sizes = np.diff(np.r_[starts, count])
    group_positives = np.add.reduceat(sorted_labels, starts)
    slot_mass = np.repeat(group_positives / sizes, sizes)
    expected_rejected_positives = np.r_[0.0, np.cumsum(slot_mass)][:-1]
    retained = count - np.arange(count, dtype=np.float64)
    precision = 1.0 - (positives - expected_rejected_positives) / retained
    return float(precision.mean())


def prr(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Prediction Rejection Ratio; 0 for random ranking, 1 for oracle ranking."""
    score_array, label_array = _validated(scores, labels)
    positives = int(label_array.sum())
    if positives == 0 or positives == label_array.size:
        return None
    area = _rejection_area(score_array, label_array)
    random_area = 1.0 - positives / label_array.size
    oracle_area = _rejection_area(label_array.astype(np.float64), label_array)
    return (area - random_area) / (oracle_area - random_area)


def ece(scores: Sequence[float], labels: Sequence[int], *, bins: int = 15) -> float:
    """Calibration gap after a label-free min-max mapping into equal-width bins."""
    if bins < 1:
        raise ValueError("bins must be positive")
    score_array, label_array = _validated(scores, labels)
    minimum = score_array.min()
    maximum = score_array.max()
    if maximum > minimum:
        normalized = (score_array - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(score_array)
    indices = np.minimum((normalized * bins).astype(np.int64), bins - 1)
    score_sums = np.bincount(indices, weights=normalized, minlength=bins)
    label_sums = np.bincount(indices, weights=label_array.astype(np.float64), minlength=bins)
    return float(np.abs(score_sums - label_sums).sum() / score_array.size)


def cluster_bootstrap_indices(
    clusters: Sequence[str], *, n_bootstrap: int, seed: int
) -> list[np.ndarray]:
    """Index arrays for bootstrap replicates that resample whole clusters."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not clusters:
        raise ValueError("clusters must be non-empty")
    members: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("cluster identifiers must be non-empty strings")
        members.setdefault(cluster, []).append(index)
    arrays = [np.asarray(indices, dtype=np.int64) for indices in members.values()]
    generator = np.random.default_rng(seed)
    replicates = []
    for _ in range(n_bootstrap):
        chosen = generator.integers(0, len(arrays), size=len(arrays))
        replicates.append(np.concatenate([arrays[index] for index in chosen]))
    return replicates


def bootstrap_summary(
    metric: Callable[[np.ndarray], float | None],
    replicates: Sequence[np.ndarray],
    *,
    confidence: float = 0.95,
) -> dict:
    """Percentile confidence interval of ``metric`` over precomputed replicates.

    Replicates on which the metric is undefined are counted and excluded
    instead of being imputed.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if not replicates:
        raise ValueError("replicates must be non-empty")
    values = [metric(replicate) for replicate in replicates]
    defined = [value for value in values if value is not None]
    undefined = len(values) - len(defined)
    if not defined:
        return {"ci_low": None, "ci_high": None, "undefined_replicates": undefined}
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(defined, (tail, 100.0 - tail))
    return {"ci_low": float(low), "ci_high": float(high), "undefined_replicates": undefined}
