"""Abstract base class for models."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class Model(ABC):
    """Interface for multimodal / text-only generation models.

    Subclasses must implement ``generate``.  The interface mirrors
    vauq-repro's ``Backend`` so the two baselines share the same
    model-loading conventions.
    """

    device: str

    @abstractmethod
    def generate(
        self,
        question: str,
        image: Image.Image | None = None,
        temperature: float = 0.1,
        max_new_tokens: int = 64,
    ) -> tuple[str, list[float], "torch.Tensor | None"]:  # noqa: F821
        """Sample one answer and return (text, token_log_likelihoods, embedding).

        Parameters
        ----------
        question:
            Natural-language question text.
        image:
            Input image (PIL RGB), or *None* for text-only models.
        temperature:
            Sampling temperature (0.1 = near-greedy).
        max_new_tokens:
            Maximum number of tokens to generate.

        Returns
        -------
        answer : str
            Decoded answer text.
        token_log_likelihoods : list[float]
            Per-token log-probability for each generated token.
        embedding : torch.Tensor | None
            Last-token hidden state from the final layer, or *None* if the
            model does not expose hidden states.
        """
        ...
