from __future__ import annotations

import math
from types import SimpleNamespace

from sem_unc.semantic_entropy import SemanticEntropyMethod, compute_semantic_entropy


class ExactEntailment:
    model_id = "exact-test"

    def check_pairs(self, pairs):
        return [premise == hypothesis for premise, hypothesis in pairs]


class RefusingEntailment:
    """Denies every pair, mimicking DeBERTa rejecting identical short answers."""

    model_id = "refusing-test"

    def __init__(self) -> None:
        self.queried_pairs = []

    def check_pairs(self, pairs):
        self.queried_pairs.extend(pairs)
        return [False for _ in pairs]


def test_one_cluster_has_zero_entropy() -> None:
    result = compute_semantic_entropy(
        "Color?", ["red", "red", "red"], [-0.1, -0.2, -0.3], ExactEntailment()
    )
    assert result.semantic_ids == (0, 0, 0)
    assert result.semantic_entropy == 0.0
    assert math.isclose(result.cluster_probs[0], 1.0)


def test_distinct_equal_probability_answers_have_log_two_entropy() -> None:
    result = compute_semantic_entropy(
        "Color?", ["red", "blue"], [-0.5, -0.5], ExactEntailment()
    )
    assert result.semantic_ids == (0, 1)
    assert math.isclose(result.semantic_entropy, math.log(2), rel_tol=1e-9)


def test_identical_answers_cluster_without_entailment_queries() -> None:
    entailment = RefusingEntailment()
    result = compute_semantic_entropy(
        "Color?", ["red", "red", "blue", "red"], [-0.5, -0.5, -0.5, -0.5], entailment
    )
    assert result.semantic_ids == (0, 0, 1, 0)
    for premise, hypothesis in entailment.queried_pairs:
        assert premise != hypothesis


def test_paraphrase_merging_still_uses_entailment() -> None:
    class RedCrimsonEntailment:
        model_id = "paraphrase-test"

        def check_pairs(self, pairs):
            equivalent = {"red", "crimson"}
            return [
                premise.rsplit("Answer: ", 1)[1] in equivalent
                and hypothesis.rsplit("Answer: ", 1)[1] in equivalent
                for premise, hypothesis in pairs
            ]

    result = compute_semantic_entropy(
        "Color?", ["red", "crimson", "crimson"], [-0.5, -0.5, -0.5], RedCrimsonEntailment()
    )
    assert result.semantic_ids == (0, 0, 0)


def test_online_method_returns_only_score_and_clustering_audit() -> None:
    method = SemanticEntropyMethod(ExactEntailment())
    result = method.compute(
        question="Color?",
        greedy=None,
        samples=[
            SimpleNamespace(answer="red", mean_log_prob=-0.5),
            SimpleNamespace(answer="blue", mean_log_prob=-0.5),
        ],
    )
    assert result["valid"] is True
    assert math.isclose(result["score"], math.log(2), rel_tol=1e-9)
    assert result["semantic_ids"] == [0, 1]
    assert result["clusters"] == [
        {"cluster_id": 0, "members": [0], "probability": 0.5},
        {"cluster_id": 1, "members": [1], "probability": 0.5},
    ]
    assert "mean_log_probs" not in result
