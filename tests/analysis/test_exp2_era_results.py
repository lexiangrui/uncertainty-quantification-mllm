"""Tests for the consolidated Experiment 2 ERA analysis."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.analysis.exp2_era_results import (
    attention_features,
    correctness_stratified_ablation,
    holm_adjust,
    percentile_interval,
    two_sided_bootstrap_p,
)


def test_attention_features_normalizes_and_excludes_answer_mass() -> None:
    payload = {
        "n_visual_tokens": 32,
        "n_heads": 2,
        "section_tokens": {"vision": 3, "reasoning": 4, "answer": 2},
        "layer_masses": {
            "0": [[0] * 5, [0] * 5, [4, 8, 12, 16, 20]],
            "1": [[0] * 5, [0] * 5, [8, 12, 16, 20, 24]],
        },
    }

    result = attention_features(payload, [0, 1])

    assert result["attn_image"] == pytest.approx(1.5)
    assert result["attn_prompt_text"] == pytest.approx(2.5)
    assert result["attn_vision"] == pytest.approx(3.5)
    assert result["attn_reasoning"] == pytest.approx(4.5)
    assert result["attn_answer"] == pytest.approx(5.5)
    assert result["attn_external"] == pytest.approx(4.0)
    assert result["attn_internal"] == pytest.approx(8.0)
    assert result["U_ERA"] == pytest.approx(((7.0 / 10.0) + (9.0 / 14.0)) / 2.0)
    assert result["U_ERA_with_answer"] == pytest.approx(((7.0 / 15.0) + (9.0 / 20.0)) / 2.0)
    assert result["n_visual_tokens"] == 32
    assert result["n_answer_tokens"] == 2
    assert result["n_rationale_tokens"] == 7


def test_attention_features_rejects_absent_layers() -> None:
    payload = {
        "n_visual_tokens": 1,
        "n_heads": 1,
        "section_tokens": {"vision": 1, "reasoning": 1, "answer": 1},
        "layer_masses": {"0": [[0] * 5, [0] * 5, [1] * 5]},
    }

    with pytest.raises(ValueError, match="selected layers are absent"):
        attention_features(payload, [1])


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, float("nan")])

    assert adjusted[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert math.isnan(adjusted[3])


def test_bootstrap_helpers_are_two_sided_and_percentile_based() -> None:
    values = [-2.0, -1.0, 1.0, 2.0]

    assert percentile_interval(values, confidence=0.5) == pytest.approx((-1.25, 1.25))
    assert two_sided_bootstrap_p(values) == pytest.approx(1.0)


def test_correctness_stratified_ablation_splits_each_answer_class(
    tmp_path: Path,
) -> None:
    records = [
        {"label": 1, "correct": True, "U_ERA": 0.9, "group_id": "a"},
        {"label": 0, "correct": True, "U_ERA": 0.1, "group_id": "b"},
        {"label": 1, "correct": False, "U_ERA": 0.8, "group_id": "c"},
        {"label": 0, "correct": False, "U_ERA": 0.2, "group_id": "d"},
    ]

    rows = correctness_stratified_ablation(
        {"synthetic": records}, n_bootstrap=20, out_dir=tmp_path
    )

    assert [(row["stratum"], row["n"]) for row in rows] == [
        ("correct", 2),
        ("incorrect", 2),
    ]
    assert all(row["n_hallucination"] == 1 for row in rows)
    assert all(row["n_non_hallucination"] == 1 for row in rows)
    assert all(row["auroc"] == pytest.approx(1.0) for row in rows)
    assert (tmp_path / "correctness_stratified_ablation.csv").exists()
