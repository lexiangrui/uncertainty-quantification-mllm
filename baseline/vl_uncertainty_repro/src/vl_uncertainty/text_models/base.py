"""Text model interfaces for rephrasing and entailment."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextModel(ABC):
    """A text-only model used by VL-Uncertainty."""

    @abstractmethod
    def generate(self, prompt: str, temp: float = 0.1, max_new_tokens: int = 256) -> str:
        raise NotImplementedError

    def generate_batch(
        self, prompts: list[str], temps: list[float], max_new_tokens: int = 256
    ) -> list[str]:
        if len(prompts) != len(temps):
            raise ValueError("prompts and temps must have the same length")
        return [
            self.generate(prompt, temp=temp, max_new_tokens=max_new_tokens)
            for prompt, temp in zip(prompts, temps)
        ]
