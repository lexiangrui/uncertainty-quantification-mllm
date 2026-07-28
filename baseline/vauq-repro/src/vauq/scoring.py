"""VAUQ scoring: predictive entropy + core visual Image-Information Score."""

from __future__ import annotations

import torch

from .metrics import OutputScoreInfo
from .types import VAUQResult

MASK_STRATEGIES = ("core", "blank", "random")


def compute_entropy(backend, image, question, generated_ids):
    with torch.no_grad():
        logits = backend.get_logits(image, question, generated_ids)
        score_info = OutputScoreInfo(logits, generated_ids, backend.device)
        return score_info.compute_entropy()


def compute_entropy_core_masked(
    backend, image, question, generated_ids, topk_ratio, layer_range,
    mask_strategy="core", mask_seed=None,
):
    with torch.no_grad():
        logits = backend.get_logits_masked(
            image,
            question,
            generated_ids,
            topk_ratio=topk_ratio,
            layer_range=tuple(layer_range),
            mask_strategy=mask_strategy,
            mask_seed=mask_seed,
        )
        score_info = OutputScoreInfo(logits, generated_ids, backend.device)
        return score_info.compute_entropy()


def compute_vauq_scores(
    backend,
    image,
    question,
    generated_ids,
    topk_ratio=0.6,
    alpha=0.5,
    layer_range=(10, 25),
    mask_strategy="core",
    mask_seed=None,
    answer=None,
) -> VAUQResult:
    """Compute VAUQ uncertainty scores for one sample.

    Returns a :class:`VAUQResult` with ``entropy`` = H(Y|X,V), ``is_score`` =
    Image-Information Score under core masking, and ``vauq`` = H - alpha * IS
    (lower VAUQ => more likely correct).
    """
    entropy_org = compute_entropy(backend, image, question, generated_ids)
    entropy_masked = compute_entropy_core_masked(
        backend, image, question, generated_ids, topk_ratio, layer_range,
        mask_strategy=mask_strategy, mask_seed=mask_seed,
    )

    is_score = entropy_masked - entropy_org
    vauq = entropy_org - alpha * is_score

    return VAUQResult(
        answer=answer,
        entropy=entropy_org,
        entropy_masked=entropy_masked,
        is_score=is_score,
        vauq=vauq,
    )


def compute_mask_comparison_scores(
    backend,
    image,
    question,
    generated_ids,
    topk_ratio=0.6,
    alpha=0.5,
    layer_range=(10, 25),
    mask_seed=None,
    answer=None,
) -> dict[str, VAUQResult]:
    """Compute core/blank/random scores from one answer and one base entropy."""
    entropy_org = compute_entropy(backend, image, question, generated_ids)
    results = {}
    for strategy in MASK_STRATEGIES:
        entropy_masked = compute_entropy_core_masked(
            backend,
            image,
            question,
            generated_ids,
            topk_ratio,
            layer_range,
            mask_strategy=strategy,
            mask_seed=mask_seed if strategy == "random" else None,
        )
        is_score = entropy_masked - entropy_org
        results[strategy] = VAUQResult(
            answer=answer,
            entropy=entropy_org,
            entropy_masked=entropy_masked,
            is_score=is_score,
            vauq=entropy_org - alpha * is_score,
        )
    return results


def compute_multi_seed_comparison_scores(
    backend,
    image,
    question,
    generated_ids,
    random_seeds,
    topk_ratio=0.6,
    alpha=0.5,
    layer_range=(10, 25),
    sample_index=0,
    answer=None,
) -> dict[str, VAUQResult]:
    """Compute core, blank and multiple random masks with one base entropy."""
    entropy_org = compute_entropy(backend, image, question, generated_ids)
    strategies = [("core", "core", None), ("blank", "blank", None)]
    strategies.extend(
        (f"random_seed{seed}", "random", seed + sample_index)
        for seed in random_seeds
    )
    results = {}
    for name, strategy, mask_seed in strategies:
        entropy_masked = compute_entropy_core_masked(
            backend,
            image,
            question,
            generated_ids,
            topk_ratio,
            layer_range,
            mask_strategy=strategy,
            mask_seed=mask_seed,
        )
        is_score = entropy_masked - entropy_org
        results[name] = VAUQResult(
            answer=answer,
            entropy=entropy_org,
            entropy_masked=entropy_masked,
            is_score=is_score,
            vauq=entropy_org - alpha * is_score,
        )
    return results
