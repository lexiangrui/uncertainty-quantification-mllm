"""Visual-token mask selection shared by VAUQ and its ablations."""

from __future__ import annotations

import torch

MASK_STRATEGIES = ("core", "random", "blank")


def choose_masked_offsets(
    num_tokens: int,
    topk_ratio: float,
    strategy: str,
    *,
    attention_scores: torch.Tensor | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Return visual-token offsets to mask, deterministically for random."""
    if strategy not in MASK_STRATEGIES:
        raise ValueError(f"unknown mask strategy {strategy!r}; choose from {MASK_STRATEGIES}")
    if not 0.0 < topk_ratio <= 1.0:
        raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}")
    if strategy == "blank":
        return torch.arange(num_tokens, dtype=torch.long)

    k = max(1, int(num_tokens * topk_ratio))
    if strategy == "core":
        if attention_scores is None or attention_scores.numel() != num_tokens:
            raise ValueError("core masking requires one attention score per visual token")
        return torch.topk(attention_scores, k).indices.detach().cpu()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0 if seed is None else seed)
    return torch.randperm(num_tokens, generator=generator)[:k]
