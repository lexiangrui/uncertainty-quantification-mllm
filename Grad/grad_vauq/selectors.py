"""Generic gradient-based visual token selectors."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .types import VisualTokenTrace


def response_nll_loss(logits: torch.Tensor, generated_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Negative log-likelihood of the fixed generated response."""

    response_logits = logits[:, prompt_len - 1 : -1, :]
    targets = generated_ids.to(response_logits.device)
    return F.cross_entropy(
        response_logits.reshape(-1, response_logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="mean",
    )


class GradXActSelector:
    """Select visual tokens by `abs(gradient * activation)`.

    This is model-agnostic as long as an adapter exposes visual embeddings that
    participate in the language-model forward pass.
    """

    name = "grad_x_act"

    def score(self, loss: torch.Tensor, trace: VisualTokenTrace) -> torch.Tensor:
        grad = torch.autograd.grad(loss, trace.features, retain_graph=False)[0]
        scores = (grad * trace.features).abs().sum(dim=-1)
        return scores[0].detach()

    def select(
        self,
        loss: torch.Tensor,
        trace: VisualTokenTrace,
        topk_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(loss, trace)
        num_tokens = scores.numel()
        k = max(1, int(num_tokens * topk_ratio))
        k = min(k, num_tokens)
        _, indices = torch.topk(scores, k)
        return indices.detach(), scores


SELECTOR_MAP = {
    GradXActSelector.name: GradXActSelector,
}


def build_selector(name: str):
    if name not in SELECTOR_MAP:
        raise ValueError(f"Unknown selector {name!r}; choose from {sorted(SELECTOR_MAP)}")
    return SELECTOR_MAP[name]()
