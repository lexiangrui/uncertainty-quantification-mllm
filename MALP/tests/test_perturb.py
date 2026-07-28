import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perturb import (
    PerturbSpec,
    apply_token_mask,
    perturb_tensor,
    sample_adversarial_gradient_noise,
    sample_directional_noise,
    sample_isotropic_noise,
    sample_norm_isotropic_noise,
)


def test_isotropic_noise_is_reproducible():
    target = torch.zeros(2, 3, 4)
    first = sample_isotropic_noise(target, seed=42)
    second = sample_isotropic_noise(target, seed=42)
    third = sample_isotropic_noise(target, seed=43)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_token_mask_keeps_unselected_tokens_unchanged():
    torch.manual_seed(0)
    target = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, True, False]])
    spec = PerturbSpec(
        modality="vision",
        stage="fusion",
        mode="directional",
        sigma=0.5,
        gamma=1.0,
        seed=42,
        token_mask=mask,
    )
    perturbed = perturb_tensor(target, spec)
    # 被 mask 排除的 token 完全不变
    assert torch.equal(perturbed[:, 1, :], target[:, 1, :])
    assert torch.equal(perturbed[:, 3, :], target[:, 3, :])
    # 被选中的 token 发生变化
    assert not torch.equal(perturbed[:, 0, :], target[:, 0, :])
    assert not torch.equal(perturbed[:, 2, :], target[:, 2, :])


def test_text_gamma_scales_only_text_modality():
    torch.manual_seed(0)
    target = torch.randn(1, 2, 3)
    vision = PerturbSpec("vision", "fusion", "directional", sigma=0.5, gamma=3.0, seed=42)
    text = PerturbSpec("text", "fusion", "directional", sigma=0.5, gamma=3.0, seed=42)
    vision_delta = perturb_tensor(target, vision) - target
    text_delta = perturb_tensor(target, text) - target
    assert torch.allclose(text_delta, vision_delta * 3.0)


def test_apply_token_mask_accepts_last_dim_mask():
    noise = torch.ones(2, 3, 4)
    mask = torch.tensor([[[1], [0], [1]], [[0], [1], [0]]], dtype=torch.bool)
    masked = apply_token_mask(noise, mask)
    assert masked[0, 1].sum().item() == 0
    assert masked[1, 0].sum().item() == 0
    assert masked[0, 0].sum().item() == 4


def test_norm_isotropic_noise_scales_with_token_norm():
    target = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -1.0, 3.0]]])
    noise = sample_norm_isotropic_noise(target, seed=42)
    assert torch.allclose(noise.norm(dim=-1), target.norm(dim=-1), atol=1e-6)


def test_norm_isotropic_perturbation_preserves_token_norm():
    target = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -1.0, 3.0]]])
    spec = PerturbSpec("vision", "fusion", "norm_isotropic", sigma=0.5, gamma=1.0, seed=42)
    perturbed = perturb_tensor(target, spec)
    assert torch.allclose(perturbed.norm(dim=-1), target.norm(dim=-1), atol=1e-6)


def test_directional_noise_is_collinear_with_target():
    target = torch.tensor([[[1.0, 2.0, 4.0], [2.0, 0.0, -1.0]]])
    noise = sample_directional_noise(target, seed=42)
    assert not torch.allclose(noise.norm(dim=-1), target.norm(dim=-1), atol=1e-6)
    for token_index in range(target.shape[1]):
        nonzero = target[0, token_index].ne(0)
        ratios = noise[0, token_index, nonzero] / target[0, token_index, nonzero]
        assert torch.allclose(ratios, ratios[:1].expand_as(ratios))


def test_adversarial_noise_follows_negative_gradient_sign():
    target = torch.ones(1, 2, 4)
    gradient = torch.tensor([[[1.0, -2.0, 0.0, 3.0], [-1.0, 0.5, -0.2, 0.0]]])
    noise = sample_adversarial_gradient_noise(target, gradient, seed=42)
    # Each non-zero-gradient token has a strictly negative dot product with
    # the log-probability gradient, so it moves in the adversarial direction.
    assert torch.all((noise.float() * gradient).sum(dim=-1) < 0)
    assert torch.equal(noise[gradient.eq(0)], torch.zeros_like(noise[gradient.eq(0)]))


def test_adversarial_noise_matches_directional_per_token_scale():
    target = torch.tensor([[[1.0, 2.0, 4.0], [2.0, 0.0, -1.0]]])
    gradient = torch.tensor([[[3.0, -1.0, 2.0], [-2.0, 4.0, 1.0]]])
    directional = sample_directional_noise(target, seed=42)
    adversarial = sample_adversarial_gradient_noise(target, gradient, seed=42)
    assert torch.allclose(
        adversarial.float().norm(dim=-1),
        directional.float().norm(dim=-1),
        atol=1e-6,
    )


def test_adversarial_zero_gradient_produces_zero_noise():
    target = torch.randn(1, 2, 4)
    noise = sample_adversarial_gradient_noise(target, torch.zeros_like(target), seed=42)
    assert torch.equal(noise, torch.zeros_like(noise))


def test_adversarial_perturbation_respects_token_mask():
    target = torch.ones(1, 3, 4)
    gradient = torch.ones_like(target)
    mask = torch.tensor([[True, False, True]])
    spec = PerturbSpec(
        modality="vision",
        stage="fusion",
        mode="adversarial",
        sigma=0.5,
        gamma=999.0,
        seed=42,
        token_mask=mask,
        adv_gradient=gradient,
    )
    perturbed = perturb_tensor(target, spec)
    delta = perturbed - target
    assert delta[:, 1, :].abs().max().item() == 0
    assert delta[:, 0, :].abs().max().item() > 0
    assert delta[:, 2, :].abs().max().item() > 0


def test_adversarial_perturbation_uses_sigma_and_text_gamma():
    target = torch.ones(1, 2, 3)
    gradient = torch.ones_like(target)
    vision = PerturbSpec(
        "vision",
        "fusion",
        "adversarial",
        sigma=0.5,
        gamma=3.0,
        seed=42,
        adv_gradient=gradient,
    )
    text = PerturbSpec(
        "text",
        "fusion",
        "adversarial",
        sigma=0.5,
        gamma=3.0,
        seed=42,
        adv_gradient=gradient,
    )
    vision_delta = perturb_tensor(target, vision) - target
    text_delta = perturb_tensor(target, text) - target
    assert torch.allclose(text_delta, vision_delta * 3.0)


def test_perturb_tensor_is_stage_agnostic():
    torch.manual_seed(0)
    target = torch.randn(1, 2, 3)
    fusion = PerturbSpec("vision", "fusion", "directional", sigma=0.5, gamma=1.0, seed=42)
    reasoning = PerturbSpec("vision", "reasoning", "directional", sigma=0.5, gamma=1.0, seed=42)
    assert torch.equal(perturb_tensor(target, fusion), perturb_tensor(target, reasoning))


if __name__ == "__main__":
    test_isotropic_noise_is_reproducible()
    test_token_mask_keeps_unselected_tokens_unchanged()
    test_text_gamma_scales_only_text_modality()
    test_apply_token_mask_accepts_last_dim_mask()
    test_norm_isotropic_noise_scales_with_token_norm()
    test_norm_isotropic_perturbation_preserves_token_norm()
    test_directional_noise_is_collinear_with_target()
    test_adversarial_noise_follows_negative_gradient_sign()
    test_adversarial_noise_matches_directional_per_token_scale()
    test_adversarial_zero_gradient_produces_zero_noise()
    test_adversarial_perturbation_respects_token_mask()
    test_adversarial_perturbation_uses_sigma_and_text_gamma()
    test_perturb_tensor_is_stage_agnostic()
    print("malp perturb tests passed")
