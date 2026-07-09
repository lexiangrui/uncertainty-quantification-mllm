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


def make_feature_baseline(features: torch.Tensor, baseline: str) -> torch.Tensor:
    if baseline == "zero":
        return torch.zeros_like(features)
    if baseline == "mean":
        return features.mean(dim=1, keepdim=True).expand_as(features).clone()
    raise ValueError("Unsupported attribution baseline. Use 'zero' or 'mean'.")


def integrated_gradients_scores(
    backend,
    image,
    question,
    generated_ids,
    features: torch.Tensor,
    baseline: str = "mean",
    steps: int = 16,
) -> torch.Tensor:
    """Approximate integrated gradients over visual token embeddings."""

    if steps <= 0:
        raise ValueError("Integrated gradients steps must be positive.")

    original = features.detach()
    start = make_feature_baseline(original, baseline)
    delta = original - start
    total_grad = torch.zeros_like(original)

    for step in range(1, steps + 1):
        scaled = (start + delta * (float(step) / float(steps))).detach()
        scaled.requires_grad_(True)
        logits, prompt_len = backend.forward_logits_with_visual_features(
            image,
            question,
            generated_ids,
            scaled,
        )
        loss = response_nll_loss(logits, generated_ids, prompt_len)
        grad = torch.autograd.grad(loss, scaled, retain_graph=False)[0]
        total_grad = total_grad + grad.detach()
        del logits, loss, grad, scaled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_grad = total_grad / float(steps)
    scores = (delta * avg_grad).abs().sum(dim=-1)
    return scores[0].detach()
