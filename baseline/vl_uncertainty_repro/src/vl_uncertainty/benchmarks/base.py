"""Benchmark interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Benchmark(ABC):
    """Pluggable multimodal benchmark."""

    benchmark_type: str = "free_form"

    @abstractmethod
    def obtain_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, idx: int) -> dict[str, Any] | None:
        """Return a dict with ``img``, ``question`` and ``gt_ans``."""
        raise NotImplementedError
