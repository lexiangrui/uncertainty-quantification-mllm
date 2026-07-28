from __future__ import annotations

import math
from types import SimpleNamespace

from umpire_uq import UmpireMethod


def signal(hidden, probability):
    return SimpleNamespace(
        final_hidden=hidden,
        sequence_log_prob=math.log(probability),
        sampling_sequence_log_prob=math.log(probability),
    )


def test_orthogonal_embeddings_and_adaptive_alpha() -> None:
    method = UmpireMethod(jitter=1e-8)
    value_a = method.compute(
        question="q",
        greedy=None,
        samples=[signal((1, 0), 0.8), signal((0, 1), 0.6)],
    )
    value_b = method.compute(
        question="q",
        greedy=None,
        samples=[signal((1, 0), 0.5), signal((1, 1), 0.5)],
    )
    method.finalize([value_a, value_b])
    assert value_a["score"] is not None
    assert value_a["alpha"] == value_b["alpha"]
    assert math.isfinite(value_a["score"])
    assert math.isclose(
        value_a["official_logdet"], 2 * math.log1p(1e-8), abs_tol=2e-16
    )
    assert math.isclose(value_a["official_incoherence_sum"], 0.6)
    assert math.isclose(
        value_a["semantic_volume"], math.log1p(1e-8) / 2, abs_tol=5e-17
    )
    assert math.isclose(value_a["incoherence_mean"], 0.3)


def test_identical_embeddings_remain_finite_with_jitter() -> None:
    method = UmpireMethod()
    value = method.compute(
        question="q",
        greedy=None,
        samples=[signal((1, 0), 0.5), signal((1, 0), 0.5)],
    )
    assert math.isfinite(value["semantic_volume"])
    assert math.isfinite(value["official_logdet"])


def test_finalize_ignores_invalid_values() -> None:
    method = UmpireMethod()
    valid = method.compute(
        question="q",
        greedy=None,
        samples=[signal((1, 0), 0.8), signal((0, 1), 0.6)],
    )
    invalid = {"valid": False, "error": "invalid response", "score": None}

    method.finalize([valid, invalid])

    assert valid["alpha"] is not None
    assert valid["score"] is not None
    assert invalid == {"valid": False, "error": "invalid response", "score": None}


def test_zero_embedding_is_rejected() -> None:
    method = UmpireMethod()

    try:
        method.compute(
            question="q",
            greedy=None,
            samples=[signal((0, 0), 0.5), signal((1, 0), 0.5)],
        )
    except ValueError as error:
        assert "non-zero norm" in str(error)
    else:
        raise AssertionError("a zero response embedding must be rejected")
