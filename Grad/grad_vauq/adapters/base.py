"""Adapter protocol for model-specific visual token access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

from ..types import VisualTokenTrace


HookFn = Callable[[torch.Tensor], torch.Tensor]


class VisionTokenAdapter(ABC):
    """Expose visual embeddings for attribution and ablation.

    Implementations should hook the tensor that represents visual tokens after
    the vision encoder/projector and immediately before those tokens influence
    the language model.
    """

    @contextmanager
    @abstractmethod
    def capture(self, model) -> Any:
        """Yield a mutable holder populated with a `VisualTokenTrace`."""

    @contextmanager
    @abstractmethod
    def ablate(self, model, indices: torch.Tensor, baseline: str = "zero") -> Any:
        """Temporarily replace selected visual tokens during model forward."""

    @contextmanager
    @abstractmethod
    def override(self, model, features: torch.Tensor) -> Any:
        """Temporarily replace all visual tokens with `features` during forward."""

    def infer_spatial_shape(self, num_tokens: int) -> tuple[int, int] | None:
        side = int(num_tokens**0.5)
        if side * side == num_tokens:
            return side, side
        return None
