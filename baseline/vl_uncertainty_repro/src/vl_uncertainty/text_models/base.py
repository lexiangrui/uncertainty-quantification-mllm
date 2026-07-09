"""Text model interfaces for rephrasing and entailment."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextModel(ABC):
    """A text-only model used by VL-Uncertainty."""

    @abstractmethod
    def generate(self, prompt: str, temp: float = 0.1, max_new_tokens: int = 256) -> str:
        raise NotImplementedError


class EntailmentModel(ABC):
    """Semantic equivalence helper."""

    @abstractmethod
    def entails(self, premise: str, hypothesis: str) -> bool:
        raise NotImplementedError
