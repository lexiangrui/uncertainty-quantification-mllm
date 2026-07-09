"""Grad-VAUQ scoring."""

from __future__ import annotations

import torch

from vauq.metrics import OutputScoreInfo

from .selectors import build_selector, response_nll_loss
from .types import GradVAUQResult


def compute_response_entropy(logits: torch.Tensor, generated_ids: torch.Tensor, prompt_len: int, device: str) -> float:
    response_logits = logits[0, prompt_len - 1 : -1]
    score_info = OutputScoreInfo(response_logits, generated_ids, device)
    return score_info.compute_entropy()


def compute_grad_vauq_scores(
    backend,
    image,
    question,
    generated_ids,
    topk_ratio: float = 0.3,
    alpha: float = 1.2,
    selector_name: str = "grad_x_act",
    ablation_baseline: str = "attention_mask",
    answer: str | None = None,
    store_visual_scores: bool = False,
) -> GradVAUQResult:
    """Compute Grad-VAUQ for one sample."""

    logits, prompt_len = backend.forward_logits(image, question, generated_ids)
    entropy_org = compute_response_entropy(logits, generated_ids, prompt_len, backend.device)
    del logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    grad_logits, grad_prompt_len, trace = backend.forward_logits_with_trace(
        image, question, generated_ids
    )
    spatial_shape = trace.spatial_shape
    loss = response_nll_loss(grad_logits, generated_ids, grad_prompt_len)
    selector = build_selector(selector_name)
    selected_indices, visual_scores = selector.select(loss, trace, topk_ratio=topk_ratio)
    del grad_logits, loss
    del trace
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    masked_logits, masked_prompt_len = backend.forward_logits_with_ablation(
        image,
        question,
        generated_ids,
        selected_indices=selected_indices,
        baseline=ablation_baseline,
    )
    entropy_masked = compute_response_entropy(
        masked_logits, generated_ids, masked_prompt_len, backend.device
    )
    del masked_logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_score = entropy_masked - entropy_org
    vauq = entropy_org - alpha * is_score

    return GradVAUQResult(
        answer=answer,
        entropy=entropy_org,
        entropy_masked=entropy_masked,
        is_score=is_score,
        vauq=vauq,
        selected_indices=[int(i) for i in selected_indices.detach().cpu().tolist()],
        visual_scores=visual_scores.detach().float().cpu().tolist() if store_visual_scores else None,
        spatial_shape=spatial_shape,
    )
