"""Tests for dual-judge reliability statistics."""

import pytest

from scripts.analysis.judge_reliability import (
    cohens_kappa,
    summarize_adjudication,
    summarize_labels,
)


def test_cohens_kappa_matches_binary_example() -> None:
    left = [True, True, False, False, True]
    right = [True, False, False, False, True]

    assert cohens_kappa(left, right) == pytest.approx(0.6153846154)


def test_summarize_labels_reports_disagreement_and_prevalence() -> None:
    rows = [
        {"gpt": {"correct": True}, "gemini": {"correct": True}},
        {"gpt": {"correct": True}, "gemini": {"correct": False}},
        {"gpt": {"correct": False}, "gemini": {"correct": False}},
    ]

    result = summarize_labels(rows, "correct")

    assert result["n"] == 3
    assert result["agreements"] == 2
    assert result["disagreements"] == 1
    assert result["agreement_rate"] == pytest.approx(2 / 3)
    assert result["gpt_positive_rate"] == pytest.approx(2 / 3)
    assert result["gemini_positive_rate"] == pytest.approx(1 / 3)


def test_summarize_adjudication_accounts_for_overlapping_fields() -> None:
    samples = [
        {"model": "llava", "disagreements": {"correct": True, "hallucination": True}},
        {"model": "llava", "disagreements": {"correct": False, "hallucination": True}},
        {"model": "qwen", "disagreements": {"correct": True, "hallucination": False}},
    ]

    overall = summarize_adjudication(samples)[-1]

    assert overall == {
        "model": "overall",
        "correctness_fields": 2,
        "hallucination_fields": 2,
        "both_fields": 1,
        "unique_samples": 3,
    }
