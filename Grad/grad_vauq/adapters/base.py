"""Adapter protocol for model-specific visual token access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any

from ..types import VisualTokenTrace


class VisionTokenAdapter(ABC):
    """Expose visual embeddings for attribution.

    Implementations should hook the tensor that represents visual tokens after
    the vision encoder/projector and immediately before those tokens influence
    the language model.
    """

    @contextmanager
    @abstractmethod
    def capture(self, model) -> Any:
        """Yield a mutable holder populated with a `VisualTokenTrace`."""

    def infer_spatial_shape(self, num_tokens: int) -> tuple[int, int] | None:
        side = int(num_tokens**0.5)
        if side * side == num_tokens:
            return side, side
        return None
