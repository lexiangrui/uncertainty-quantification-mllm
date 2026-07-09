"""Benchmark abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Benchmark(ABC):
    """Pluggable evaluation dataset.

    ``retrieve`` returns a dict with at least ``img`` (PIL), ``question`` (str)
    and ``gt_ans`` (str), plus any extra fields the judge may need (e.g.
    ``choices``). Return ``None`` if a sample should be skipped.
    """

    @abstractmethod
    def obtain_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, idx: int) -> dict[str, Any] | None:
        raise NotImplementedError
