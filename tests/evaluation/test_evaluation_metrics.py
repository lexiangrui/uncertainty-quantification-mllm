from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    auprc,
    auroc,
    bootstrap_summary,
    cluster_bootstrap_indices,
    ece,
    prr,
)


def test_auroc_counts_ties_as_half() -> None:
    assert auroc([3.0, 2.0, 2.0, 1.0], [1, 1, 0, 0]) == pytest.approx(0.875)


def test_auroc_perfect_reversed_and_constant() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert auroc([1.0, 1.0, 1.0, 1.0], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_auprc_matches_stepwise_average_precision() -> None:
    assert auprc([0.9, 0.8, 0.7], [1, 0, 1]) == pytest.approx(5.0 / 6.0)
    assert auprc([0.9, 0.8, 0.1], [1, 1, 0]) == pytest.approx(1.0)


def test_auprc_collapses_tied_scores_into_one_step() -> None:
    assert auprc([1.0, 1.0], [1, 0]) == pytest.approx(0.5)


def test_prr_is_one_for_oracle_and_zero_for_constant() -> None:
    assert prr([3.0, 2.0, 1.0], [1, 0, 0]) == pytest.approx(1.0)
    assert prr([1.0, 1.0, 1.0, 1.0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_prr_is_negative_for_anti_oracle_ranking() -> None:
    assert prr([1.0, 2.0], [1, 0]) == pytest.approx(-1.0)


def test_single_class_targets_are_undefined() -> None:
    for metric in (auroc, auprc, prr):
        assert metric([0.4, 0.6], [1, 1]) is None
        assert metric([0.4, 0.6], [0, 0]) is None


def test_ece_uses_min_max_normalization_and_equal_width_bins() -> None:
    assert ece([0.0, 0.2, 0.8, 1.0], [0, 0, 1, 1], bins=2) == pytest.approx(0.1)


def test_ece_with_constant_scores_maps_to_zero_and_stays_defined() -> None:
    assert ece([2.0, 2.0, 2.0, 2.0], [1, 0, 0, 1]) == pytest.approx(0.5)


def test_ece_places_maximum_score_in_last_bin() -> None:
    assert ece([0.0, 1.0], [0, 1], bins=15) == pytest.approx(0.0)


def test_metrics_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        auroc([float("nan"), 1.0], [0, 1])
    with pytest.raises(ValueError, match="0 and 1"):
        auroc([0.1, 0.2], [0, 2])
    with pytest.raises(ValueError, match="equally sized"):
        auroc([0.1, 0.2], [0])
    with pytest.raises(ValueError, match="non-empty"):
        auroc([], [])
    with pytest.raises(ValueError, match="bins"):
        ece([0.1, 0.2], [0, 1], bins=0)


def test_cluster_bootstrap_keeps_clusters_together() -> None:
    replicates = cluster_bootstrap_indices(
        ["a", "a", "b", "c"], n_bootstrap=50, seed=3
    )
    assert len(replicates) == 50
    for replicate in replicates:
        counts = {index: int((replicate == index).sum()) for index in range(4)}
        assert counts[0] == counts[1]
        assert counts[0] + counts[2] + counts[3] == 3
        assert replicate.size in {3, 4, 5, 6}


def test_cluster_bootstrap_is_deterministic_per_seed() -> None:
    first = cluster_bootstrap_indices(["a", "b", "c"], n_bootstrap=5, seed=11)
    second = cluster_bootstrap_indices(["a", "b", "c"], n_bootstrap=5, seed=11)
    other = cluster_bootstrap_indices(["a", "b", "c"], n_bootstrap=5, seed=12)
    assert all(np.array_equal(x, y) for x, y in zip(first, second))
    assert any(not np.array_equal(x, y) for x, y in zip(first, other))


def test_bootstrap_summary_counts_undefined_replicates() -> None:
    labels = np.array([1, 0])
    replicates = cluster_bootstrap_indices(["a", "b"], n_bootstrap=200, seed=0)
    summary = bootstrap_summary(
        lambda indices: auroc(np.array([0.9, 0.1])[indices], labels[indices]),
        replicates,
    )
    assert summary["undefined_replicates"] > 0
    assert summary["ci_low"] == pytest.approx(1.0)
    assert summary["ci_high"] == pytest.approx(1.0)


def test_bootstrap_summary_with_only_undefined_replicates() -> None:
    summary = bootstrap_summary(
        lambda indices: None, [np.array([0]), np.array([1])]
    )
    assert summary == {"ci_low": None, "ci_high": None, "undefined_replicates": 2}
