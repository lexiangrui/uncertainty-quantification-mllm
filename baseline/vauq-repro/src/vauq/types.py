from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VAUQResult:
    """Per-sample VAUQ scores."""

    answer: str | None
    entropy: float            # H(y | v, t): predictive entropy on the original image
    entropy_masked: float     # H(y | v_masked, t): entropy after core visual masking
    is_score: float           # Image-Information Score = entropy_masked - entropy
    vauq: float               # uncertainty score = entropy - alpha * is_score
