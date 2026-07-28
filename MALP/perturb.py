from dataclasses import dataclass
from typing import Literal

import torch


Modality = Literal["vision", "text", "joint"]
Stage = Literal["fusion", "reasoning", "final_hidden"]
PerturbMode = Literal["norm_isotropic", "directional", "adversarial"]


@dataclass(frozen=True)
class PerturbSpec:
    modality: Modality
    stage: Stage
    mode: PerturbMode
    sigma: float
    gamma: float
    seed: int
    token_mask: torch.Tensor | None = None
    adv_gradient: torch.Tensor | None = None


def effective_sigma(spec: PerturbSpec) -> float:
    if spec.modality == "text":
        return spec.sigma * spec.gamma
    return spec.sigma


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    # 直接使用目标张量所在设备创建 generator；当前实验在 CUDA 上运行。
    return torch.Generator(device=device).manual_seed(seed)


def sample_isotropic_noise(target: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = _make_generator(target.device, seed)
    return torch.randn(
        target.shape,
        generator=generator,
        device=target.device,
        dtype=target.dtype,
    )


def normalize_noise_to_token_norm(noise: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_norm = target.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    noise_norm = noise.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (noise.float() * (target_norm / noise_norm)).to(dtype=target.dtype)


def preserve_token_norm(perturbed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_norm = target.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    perturbed_norm = perturbed.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (perturbed.float() * (target_norm / perturbed_norm)).to(dtype=target.dtype)


def sample_norm_isotropic_noise(target: torch.Tensor, *, seed: int) -> torch.Tensor:
    return normalize_noise_to_token_norm(sample_isotropic_noise(target, seed=seed), target)


def sample_directional_noise(target: torch.Tensor, *, seed: int) -> torch.Tensor:
    # 在 fp32 下采样标量系数再 cast 回 target.dtype，避免 fp16 直接采样带来的精度损失。
    generator = _make_generator(target.device, seed)
    scalar_shape = (*target.shape[:-1], 1)
    coefficients = torch.randn(
        scalar_shape,
        generator=generator,
        device=target.device,
        dtype=torch.float32,
    ).to(dtype=target.dtype)
    return target * coefficients


def sample_adversarial_gradient_noise(
    target: torch.Tensor,
    gradient: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    if gradient.shape != target.shape:
        raise AssertionError(
            f"adversarial gradient shape {tuple(gradient.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    generator = _make_generator(target.device, seed)
    scalar_shape = (*target.shape[:-1], 1)
    step = torch.randn(
        scalar_shape,
        generator=generator,
        device=target.device,
        dtype=torch.float32,
    ).abs()
    # Match directional noise's per-token magnitude exactly:
    #   ||alpha * target||_2 = |alpha| * ||target||_2.
    # Only the direction changes, from the token vector to the negative
    # gradient unit vector (which lowers the fixed answer log-probability).
    target_norm = target.float().norm(dim=-1, keepdim=True)
    gradient_float = gradient.float()
    # Scale before the L2 norm so very large sanitized fp16 gradients cannot
    # overflow while constructing the unit direction.
    gradient_scale = gradient_float.abs().amax(dim=-1, keepdim=True)
    scaled_gradient = torch.where(
        gradient_scale > 0,
        gradient_float / gradient_scale.clamp_min(1e-12),
        torch.zeros_like(gradient_float),
    )
    gradient_norm = scaled_gradient.norm(dim=-1, keepdim=True)
    gradient_direction = torch.where(
        gradient_norm > 0,
        scaled_gradient / gradient_norm.clamp_min(1e-12),
        torch.zeros_like(scaled_gradient),
    )
    noise = -step * target_norm * gradient_direction
    return noise.to(dtype=target.dtype)


def apply_token_mask(noise: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
    if token_mask is None:
        return noise
    mask = token_mask.to(device=noise.device, dtype=noise.dtype)
    while mask.ndim < noise.ndim:
        mask = mask.unsqueeze(-1)
    return noise * mask


def perturb_tensor(target: torch.Tensor, spec: PerturbSpec) -> torch.Tensor:
    if spec.mode == "norm_isotropic":
        # 先把随机方向缩放到原 token 范数，加扰动后再投影回原范数。
        # 被 token_mask 排除的位置噪声为 0，preserve_token_norm(target, target) == target，
        # 因此无需再做 blend。
        noise = sample_norm_isotropic_noise(target, seed=spec.seed)
        noise = apply_token_mask(noise, spec.token_mask)
        return preserve_token_norm(target + effective_sigma(spec) * noise, target)
    elif spec.mode == "directional":
        noise = sample_directional_noise(target, seed=spec.seed)
    elif spec.mode == "adversarial":
        if spec.adv_gradient is None:
            raise ValueError("adversarial perturbation requires spec.adv_gradient")
        noise = sample_adversarial_gradient_noise(
            target,
            spec.adv_gradient.to(device=target.device),
            seed=spec.seed,
        )
    else:
        raise ValueError(f"unknown perturbation mode: {spec.mode}")
    noise = apply_token_mask(noise, spec.token_mask)
    return target + effective_sigma(spec) * noise
