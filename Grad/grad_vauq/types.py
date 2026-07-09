"""Shared dataclasses for Grad-VAUQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class VisualTokenTrace:
    """Visual embeddings captured at the adapter boundary.

    `features` should be a leaf tensor with `requires_grad=True` when used for
    attribution. Shape is expected to be `[batch, num_visual_tokens, hidden]`.
    """

    features: torch.Tensor
    spatial_shape: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GradVAUQResult:
    """Per-sample Grad-VAUQ scores and attribution metadata."""

    answer: str | None
    entropy: float
    entropy_masked: float
    is_score: float
    vauq: float
    selected_indices: list[int]
    visual_scores: list[float] | None = None
    spatial_shape: tuple[int, int] | None = None
