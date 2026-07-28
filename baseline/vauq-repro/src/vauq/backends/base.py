"""Backend abstract base class.

A backend wraps a local white-box LVLM and exposes the three forward passes
that VAUQ needs: sampled generation, teacher-forced logits over the original
image, and teacher-forced logits after core visual masking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    """Pluggable LVLM backend.

    Implementations must expose ``self.device`` (the device logits/ids live on)
    and the three methods below. ``image`` is a PIL image, ``question`` a str,
    ``generated_ids`` a ``[1, L]`` int tensor on ``self.device``.
    """

    device: str

    @abstractmethod
    def generate_with_ids(self, image, question, max_new_tokens: int = 128):
        """Return ``(answer: str, generated_ids: Tensor[1, L])``."""
        raise NotImplementedError

    @abstractmethod
    def get_logits(self, image, question, generated_ids):
        """Teacher-forced forward over ``[prompt; generated_ids]``.

        Return ``logits[0, prompt_len - 1 : -1]`` of shape ``[L, vocab]``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_logits_masked(
        self,
        image,
        question,
        generated_ids,
        topk_ratio: float,
        layer_range: tuple[int, int],
        mask_strategy: str = "core",
        mask_seed: int | None = None,
    ):
        """Teacher-forced logits after visual-token masking.

        ``core`` masks the attention-ranked top-k tokens, ``random`` masks the
        same number uniformly without replacement, and ``blank`` masks every
        visual token.  Return ``logits[0, prompt_len - 1 : -1]``.
        """
        raise NotImplementedError
