"""Shared data structures for VL-Uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VLUncertaintyResult:
    """Per-sample uncertainty result."""

    uncertainty: float
    cluster_ids: list[int]
    cluster_distribution: dict[str, Any]  # str → float/int
    sampled_answers: list[str]
    perturbed_questions: list[str]
    most_likely_answer: str = ""
