"""Multimodal model backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

class Backend(ABC):
    """Pluggable LVLM backend used for answer generation."""

    device: str

    @abstractmethod
    def generate(
        self,
        image,
        question: str,
        temp: float = 0.1,
        max_new_tokens: int = 64,
    ) -> str:
        """Sample one answer."""
        ...

    def generate_batch(
        self,
        images: list,
        questions: list[str],
        temp: float = 0.1,
        max_new_tokens: int = 64,
    ) -> list[str]:
        if len(images) != len(questions):
            raise ValueError("images and questions must have the same length")
        return [
            self.generate(image, question, temp=temp, max_new_tokens=max_new_tokens)
            for image, question in zip(images, questions)
        ]
