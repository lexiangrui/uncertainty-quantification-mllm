import math

import torch


PERTURBATION_MODES = ("replace", "norm_isotropic")


def select_sensitive_indices(
    gradient_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """Select the largest answer-NLL gradient norms within one modality."""
    if gradient_scores.ndim != 1 or candidate_mask.shape != gradient_scores.shape:
        raise ValueError("scores and candidate mask must be equally sized vectors")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate mask must be boolean")
    if not torch.isfinite(gradient_scores).all():
        raise ValueError("gradient scores must be finite")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("selection ratio must be in (0, 1]")
    candidates = torch.nonzero(candidate_mask, as_tuple=True)[0]
    if candidates.numel() == 0:
        raise ValueError("candidate mask is empty")
    count = max(1, math.ceil(candidates.numel() * ratio))
    local_order = torch.argsort(gradient_scores[candidates], descending=True, stable=True)
    return candidates[local_order[:count]]


def replace_with_gaussian(
    embeddings: torch.Tensor,
    selected_indices: torch.Tensor,
    reference_indices: torch.Tensor,
    *,
    seed: int,
    scale: float,
) -> torch.Tensor:
    """Replace selected pre-attention vectors with zero-mean, scale-matched noise."""
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise ValueError("embedding replacement currently requires a batch of one")
    if selected_indices.ndim != 1 or reference_indices.ndim != 1:
        raise ValueError("selected and reference indices must be vectors")
    if selected_indices.numel() == 0 or reference_indices.numel() == 0:
        raise ValueError("selected and reference indices must be non-empty")
    if scale <= 0.0:
        raise ValueError("Gaussian noise scale must be positive")
    selected_indices = selected_indices.to(embeddings.device)
    reference_indices = reference_indices.to(embeddings.device)
    if int(selected_indices.max()) >= embeddings.shape[1] or int(reference_indices.max()) >= embeddings.shape[1]:
        raise IndexError("embedding replacement index is out of range")

    reference = embeddings[:, reference_indices, :].float()
    rms = reference.square().mean().sqrt().clamp_min(torch.finfo(torch.float32).eps)
    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    noise = torch.randn(
        (1, selected_indices.numel(), embeddings.shape[-1]),
        generator=generator,
        device=embeddings.device,
        dtype=embeddings.dtype,
    ) * (rms * scale).to(embeddings.dtype)
    changed = embeddings.clone()
    changed[:, selected_indices, :] = noise
    return changed


def add_norm_preserving_gaussian(
    embeddings: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    seed: int,
    sigma: float,
) -> torch.Tensor:
    """Add isotropic Gaussian noise, then restore each selected token's norm."""
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise ValueError("embedding perturbation currently requires a batch of one")
    if selected_indices.ndim != 1 or selected_indices.numel() == 0:
        raise ValueError("selected indices must be a non-empty vector")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("norm-isotropic sigma must be finite and positive")
    selected_indices = selected_indices.to(embeddings.device)
    if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= embeddings.shape[1]:
        raise IndexError("embedding perturbation index is out of range")

    target = embeddings[:, selected_indices, :]
    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    raw_noise = torch.randn(
        target.shape,
        generator=generator,
        device=target.device,
        dtype=target.dtype,
    )
    target_norm = target.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    temporary = target + sigma * raw_noise
    temporary_norm = temporary.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    perturbed = (temporary.float() * (target_norm / temporary_norm)).to(target.dtype)

    changed = embeddings.clone()
    changed[:, selected_indices, :] = perturbed
    return changed


def perturb_embeddings(
    embeddings: torch.Tensor,
    selected_indices: torch.Tensor,
    reference_indices: torch.Tensor,
    *,
    mode: str,
    seed: int,
    replacement_scale: float,
    norm_isotropic_sigma: float,
) -> torch.Tensor:
    if mode == "replace":
        return replace_with_gaussian(
            embeddings,
            selected_indices,
            reference_indices,
            seed=seed,
            scale=replacement_scale,
        )
    if mode == "norm_isotropic":
        return add_norm_preserving_gaussian(
            embeddings,
            selected_indices,
            seed=seed,
            sigma=norm_isotropic_sigma,
        )
    raise ValueError(f"unknown perturbation mode: {mode!r}")


def nll_instability(base_nll: float, perturbed_nlls: list[float]) -> dict[str, float]:
    """Aggregate fixed-answer NLL changes from repeated embedding replacements."""
    if not perturbed_nlls:
        raise ValueError("at least one perturbed NLL is required")
    deltas = torch.tensor(perturbed_nlls, dtype=torch.float64) - base_nll
    if not torch.isfinite(deltas).all():
        raise ValueError("NLL values must be finite")
    mean_absolute_delta = float(deltas.abs().mean().item())
    return {
        "score": 1.0 - math.exp(-mean_absolute_delta),
        "mean_delta": float(deltas.mean().item()),
        "mean_absolute_delta": mean_absolute_delta,
        "std_delta": float(deltas.std(unbiased=False).item()),
    }


def semantic_volume(hidden_vectors: list[torch.Tensor], jitter: float) -> dict:
    """Compute an UMPIRE-style semantic volume from normalized hidden vectors."""
    if len(hidden_vectors) < 2:
        raise ValueError("at least two hidden vectors are required")
    if not math.isfinite(jitter) or jitter <= 0.0:
        raise ValueError("semantic-volume jitter must be finite and positive")
    if any(vector.ndim != 1 for vector in hidden_vectors):
        raise ValueError("hidden vectors must be one-dimensional")
    if len({vector.numel() for vector in hidden_vectors}) != 1:
        raise ValueError("hidden vectors must have the same size")
    matrix = torch.stack([vector.float().cpu() for vector in hidden_vectors])
    if not torch.isfinite(matrix).all():
        raise ValueError("hidden vectors must be finite")
    norms = matrix.norm(p=2, dim=-1)
    if (norms <= torch.finfo(matrix.dtype).eps).any():
        raise ValueError("cannot compute cosine similarity for a zero hidden vector")
    normalized = matrix / norms[:, None]
    gram = (normalized.double() @ normalized.double().T)
    gram = 0.5 * (gram + gram.T)
    sample_count = len(hidden_vectors)
    regularized = gram + jitter * torch.eye(sample_count, dtype=gram.dtype)
    sign, logabsdet = torch.linalg.slogdet(regularized)
    if float(sign.item()) <= 0.0 or not torch.isfinite(logabsdet):
        raise FloatingPointError("regularized semantic Gram matrix is not positive definite")
    eigenvalues = torch.linalg.eigvalsh(regularized).clamp_min(jitter)
    raw_log_volume = float((logabsdet / (2.0 * sample_count)).item())

    # With unit-norm rows, identical samples give the minimum reference volume
    # (one eigenvalue K+jitter and K-1 eigenvalues jitter), while orthogonal
    # samples give the maximum reference volume (all eigenvalues 1+jitter).
    minimum = (
        math.log(sample_count + jitter) + (sample_count - 1) * math.log(jitter)
    ) / (2.0 * sample_count)
    maximum = 0.5 * math.log1p(jitter)
    normalized_volume = min(max((raw_log_volume - minimum) / (maximum - minimum), 0.0), 1.0)
    return {
        "score": normalized_volume,
        "log_volume": raw_log_volume,
        "reference_min_log_volume": minimum,
        "reference_max_log_volume": maximum,
        "eigenvalues": eigenvalues.tolist(),
        "cosine_gram_matrix": gram.tolist(),
    }


def visual_dependency_scores(nll_dependency: float, volume_dependency: float) -> dict[str, float]:
    """Convert visual response changes into support and ungrounded-answer risk.

    Large changes after removing visually sensitive embeddings mean that the
    original answer depended on visual evidence, so they lower ungrounded risk.
    """
    values = (nll_dependency, volume_dependency)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("visual dependency components must be finite values in [0, 1]")
    dependency = 0.5 * (nll_dependency + volume_dependency)
    return {
        "visual_dependency": dependency,
        "visual_ungrounded_risk": 1.0 - dependency,
    }


def combined_uncertainty(predictive_uncertainty: float, visual_ungrounded_risk: float) -> float:
    """Fuse low confidence and missing visual support with a noisy-OR.

    Either channel can flag an error. Stronger visual dependency lowers the
    visual risk and therefore monotonically lowers the combined uncertainty
    when predictive uncertainty is held fixed.
    """
    values = (predictive_uncertainty, visual_ungrounded_risk)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("uncertainty components must be finite values in [0, 1]")
    return 1.0 - (1.0 - predictive_uncertainty) * (1.0 - visual_ungrounded_risk)
