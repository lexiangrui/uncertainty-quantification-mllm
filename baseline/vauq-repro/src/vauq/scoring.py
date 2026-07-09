"""VAUQ scoring: predictive entropy + Image-Information Score (IS).

Two masking strategies:

- ``"core"`` (default): mask the top-k_ratio most-attended visual tokens via
  ``backend.get_logits_masked`` (attention-mask knockout).
- ``"blank"``: feed a blank image to ``backend.get_logits``. Fallback for
  backends whose core-attention knockout is unsupported on the current runtime.
"""

from __future__ import annotations

import torch

from .metrics import OutputScoreInfo
from .types import VAUQResult


def compute_entropy(backend, image, question, generated_ids):
    with torch.no_grad():
        logits = backend.get_logits(image, question, generated_ids)
        score_info = OutputScoreInfo(logits, generated_ids, backend.device)
        return score_info.compute_entropy()


def compute_entropy_core_masked(
    backend, image, question, generated_ids, topk_ratio, layer_range, ablation_baseline="attention_mask"
):
    with torch.no_grad():
        logits = backend.get_logits_masked(
            image,
            question,
            generated_ids,
            topk_ratio=topk_ratio,
            layer_range=tuple(layer_range),
            ablation_baseline=ablation_baseline,
        )
        score_info = OutputScoreInfo(logits, generated_ids, backend.device)
        return score_info.compute_entropy()


def compute_entropy_blank(backend, image, question, generated_ids):
    from .images import blank_like

    blank = blank_like(image) if image is not None else None
    return compute_entropy(backend, blank, question, generated_ids)


def compute_vauq_scores(
    backend,
    image,
    question,
    generated_ids,
    topk_ratio=0.6,
    alpha=0.5,
    layer_range=(10, 25),
    mask_strategy="core",
    ablation_baseline="attention_mask",
    answer=None,
) -> VAUQResult:
    """Compute VAUQ uncertainty scores for one sample.

    Returns a :class:`VAUQResult` with ``entropy`` = H(Y|X,V), ``is_score`` =
    Image-Information Score under core masking, and ``vauq`` = H - alpha * IS
    (lower VAUQ => more likely correct).
    """
    entropy_org = compute_entropy(backend, image, question, generated_ids)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if mask_strategy == "blank":
        entropy_masked = compute_entropy_blank(
            backend, image, question, generated_ids
        )
    else:
        entropy_masked = compute_entropy_core_masked(
            backend, image, question, generated_ids, topk_ratio, layer_range, ablation_baseline
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_score = entropy_masked - entropy_org
    vauq = entropy_org - alpha * is_score

    return VAUQResult(
        answer=answer,
        entropy=entropy_org,
        entropy_masked=entropy_masked,
        is_score=is_score,
        vauq=vauq,
    )
