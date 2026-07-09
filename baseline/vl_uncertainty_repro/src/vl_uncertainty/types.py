"""Shared data structures for VL-Uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    """Single generation with token-level information."""

    answer: str
    token_log_likelihoods: list[float] = field(default_factory=list)
    embedding: Any = None  # torch.Tensor | None


@dataclass
class VLUncertaintyResult:
    """Per-sample uncertainty result."""

    uncertainty: float
    cluster_ids: list[int]
    cluster_distribution: dict[str, Any]  # str → float/int
    sampled_answers: list[str]
    perturbed_questions: list[str]
    entailment: dict[str, Any] = field(default_factory=dict)
    most_likely_answer: str = ""
    most_likely_log_liks: list[float] = field(default_factory=list)


@dataclass
class Generation:
    """Text generation result from a model backend."""

    text: str
