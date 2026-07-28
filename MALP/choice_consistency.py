"""Metrics for uncertainty from perturbed multiple-choice generations."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


INVALID_CHOICE = "INVALID"


def choice_classes(texts: list[str], num_choices: int) -> list[str]:
    """Map generated texts to option letters, retaining parse failures."""
    from judge.choice import extract_choice_letter

    return [extract_choice_letter(text, num_choices) or INVALID_CHOICE for text in texts]


def consistency_metrics(classes: list[str], base_class: str) -> dict[str, Any]:
    """Compute entropy and disagreement metrics over K answer classes."""
    if len(classes) < 2:
        raise ValueError("answer consistency requires at least two perturbed generations")
    counts = Counter(classes)
    total = len(classes)
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    normalized_entropy = entropy / math.log(total)
    variation_ratio = 1.0 - max(counts.values()) / total
    base_flip_rate = sum(value != base_class for value in classes) / total
    disagreeing_pairs = sum(
        count_a * count_b
        for index, count_a in enumerate(counts.values())
        for count_b in list(counts.values())[index + 1 :]
    )
    pair_count = total * (total - 1) / 2
    return {
        "class_counts": dict(sorted(counts.items())),
        "num_answer_classes": len(counts),
        "answer_entropy": normalized_entropy,
        "variation_ratio": variation_ratio,
        "base_flip_rate": base_flip_rate,
        "pairwise_disagreement": disagreeing_pairs / pair_count,
        "invalid_rate": counts.get(INVALID_CHOICE, 0) / total,
    }
