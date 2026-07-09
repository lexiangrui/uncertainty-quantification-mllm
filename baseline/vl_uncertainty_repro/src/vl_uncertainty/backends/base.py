"""Multimodal model backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Backend(ABC):
    """Pluggable LVLM backend.

    ``generate`` returns token log-likelihoods so that semantic uncertainty
    can weight clusters by model confidence (Faithful to Farquhar et al.).
    """

    device: str

    @abstractmethod
    def generate(
        self,
        image,
        question: str,
        temp: float = 0.1,
        max_new_tokens: int = 64,
    ) -> tuple[str, list[float], torch.Tensor | None]:
        """Sample one answer.

        Returns
        -------
        answer : str
            Decoded answer text.
        token_log_likelihoods : list[float]
            Per-token log-probabilities for each generated token.
        last_token_embedding : torch.Tensor | None
            Last-token hidden state from the final layer, or *None*.
        """
        ...
