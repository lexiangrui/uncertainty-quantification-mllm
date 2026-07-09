"""Correctness judge interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Judge(ABC):
    @abstractmethod
    def judge(self, prediction: str, gold: Any, sample: dict[str, Any] | None = None) -> bool | None:
        raise NotImplementedError
