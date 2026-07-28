#!/usr/bin/env python3
"""Sweep norm-preserving perturbations over post-projector visual embeddings."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import LLAVA_MODEL, PERTURB_ROOT, SEED  # noqa: E402
from data import iter_samples  # noqa: E402
from io_utils import append_jsonl, load_jsonl_by_id  # noqa: E402
from model import LlavaMalpRunner  # noqa: E402
from perturb import PerturbSpec, perturb_tensor  # noqa: E402
from visual_embedding_sweep import (  # noqa: E402
    normalized_answer,
    summarize_trials,
    validate_sigmas,
)


LOGGER = logging.getLogger("malp.run_visual_embedding_sweep")
DEFAULT_SIGMAS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Increase a norm-preserving random perturbation after the visual projector "
            "and measure answer/NLL/KL instability."
        )
    )
    parser.add_argument(
        "--dataset", required=True, choices=["cvbench", "cvbench2d", "mmvet", "vilp"]
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMAS))
    parser.add_argument("--num-directions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output", type=Path, default=PERTURB_ROOT / "visual_embedding_sweep.jsonl"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def perturbation_geometry(original: torch.Tensor, perturbed: torch.Tensor) -> dict:
    original_float = original.float()
    perturbed_float = perturbed.float()
    original_norm = original_float.norm(dim=-1)
    perturbed_norm = perturbed_float.norm(dim=-1)
    valid = original_norm > 1e-12
    relative_l2 = (perturbed_float - original_float).norm(dim=-1) / original_norm.clamp_min(1e-12)
    cosine = F.cosine_similarity(original_float, perturbed_float, dim=-1)
    norm_relative_error = (perturbed_norm - original_norm).abs() / original_norm.clamp_min(1e-12)
    if not valid.any():
        raise RuntimeError("post-projector visual embeddings have zero norm")
    return {
        "num_visual_tokens": int(valid.sum().item()),
        "mean_token_relative_l2": float(relative_l2[valid].mean().item()),
        "mean_token_cosine": float(cosine[valid].mean().item()),
        "max_token_norm_relative_error": float(norm_relative_error[valid].max().item()),
    }


def install_post_projector_hook(
    runner: LlavaMalpRunner, spec: PerturbSpec, geometry: dict
) -> torch.utils.hooks.RemovableHandle:
    """Perturb only the projector output; text embeddings remain untouched."""

    def hook(_module, _args, output):
        if not torch.is_tensor(output):
            raise TypeError("multi_modal_projector output must be a tensor")
        changed = perturb_tensor(output, spec)
        if not geometry:
            geometry.update(perturbation_geometry(output.detach(), changed.detach()))
        return changed

    return runner.projector.register_forward_hook(hook)


@torch.inference_mode()
def generate_perturbed(
    runner: LlavaMalpRunner, inputs: dict[str, torch.Tensor], spec: PerturbSpec
) -> tuple[dict, dict]:
    geometry: dict = {}
    handle = install_post_projector_hook(runner, spec, geometry)
    try:
        result = runner.greedy_generate(inputs)
    finally:
        handle.remove()
    if not geometry:
        raise RuntimeError("projector hook was not called during generation")
    return result, geometry


@torch.inference_mode()
def score_perturbed(
    runner: LlavaMalpRunner,
    teacher_inputs: dict[str, torch.Tensor | int],
    spec: PerturbSpec,
) -> torch.Tensor:
    geometry: dict = {}
    handle = install_post_projector_hook(runner, spec, geometry)
    try:
        logits = runner.forward_original(teacher_inputs)["response_logits"]
    finally:
        handle.remove()
    if not geometry:
        raise RuntimeError("projector hook was not called during teacher forcing")
    return logits


def process_one(
    sample: dict,
    runner: LlavaMalpRunner,
    *,
    sigmas: tuple[float, ...],
    seeds: tuple[int, ...],
    experiment_config: dict,
) -> dict:
    started = time.perf_counter()
    inputs = runner.prepare_inputs(sample["image"], sample["question"])
    base = runner.greedy_generate(inputs)
    teacher_inputs = runner.build_teacher_forcing_inputs(
        inputs, base["answer_ids"], base["answer_mask"]
    )
    original_logits = runner.forward_original(teacher_inputs)["response_logits"]
    original_nll = runner.mean_nll(
        original_logits, teacher_inputs["answer_ids"], teacher_inputs["answer_mask"]
    )
    base_normalized = normalized_answer(base["text"])

    trials = []
    # A seed identifies one random ray. Reusing it at every sigma isolates the
    # effect of increasing radius instead of resampling direction at each point.
    for sigma in sigmas:
        for seed in seeds:
            spec = PerturbSpec(
                modality="vision",
                stage="fusion",
                mode="norm_isotropic",
                sigma=sigma,
                gamma=1.0,
                seed=seed,
            )
            generated, geometry = generate_perturbed(runner, inputs, spec)
            perturbed_logits = score_perturbed(runner, teacher_inputs, spec)
            perturbed_nll = runner.mean_nll(
                perturbed_logits,
                teacher_inputs["answer_ids"],
                teacher_inputs["answer_mask"],
            )
            trials.append(
                {
                    "sigma": sigma,
                    "seed": seed,
                    "answer": generated["text"],
                    "answer_changed": normalized_answer(generated["text"]) != base_normalized,
                    "nll": perturbed_nll,
                    "delta_nll": perturbed_nll - original_nll,
                    "kl": runner.mean_kl(
                        original_logits, perturbed_logits, teacher_inputs["answer_mask"]
                    ),
                    "geometry": geometry,
                }
            )

    uncertainty = summarize_trials(trials, sigmas, seeds)
    return {
        "id": sample["id"],
        "dataset": sample["dataset"],
        "question": sample["question"],
        "references": sample["references"],
        "choices": sample["choices"],
        "answer_index": sample["answer_index"],
        "original_answer": base["text"],
        "original_nll": original_nll,
        "perturbation_location": "multi_modal_projector output (visual tokens only)",
        "trials": trials,
        "uncertainty": uncertainty,
        "metadata": sample["metadata"],
        "runtime_seconds": time.perf_counter() - started,
        "experiment_config": experiment_config,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sigmas = validate_sigmas(args.sigmas)
    if args.num_directions < 1:
        raise ValueError("num-directions must be positive")
    seeds = tuple(args.seed + index for index in range(args.num_directions))
    experiment_config = {
        "schema_version": 1,
        "uncertainty_method": "post_projector_visual_norm_preserving_radius_sweep",
        "dataset": args.dataset,
        "split": args.split,
        "model": str(LLAVA_MODEL),
        "location": "post_projector",
        "modality": "vision",
        "mode": "norm_isotropic",
        "same_direction_across_sigmas": True,
        "sigmas": list(sigmas),
        "seeds": list(seeds),
    }
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {args.output}")
    existing = load_jsonl_by_id(args.output) if args.resume else {}
    for record_id, record in existing.items():
        if record.get("experiment_config") != experiment_config:
            raise ValueError(f"resume configuration mismatch for {record_id!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner = LlavaMalpRunner()
    with args.output.open("a", encoding="utf-8") as handle:
        for sample in iter_samples(args.dataset, split=args.split, limit=args.limit):
            if sample["id"] in existing:
                continue
            record = process_one(
                sample,
                runner,
                sigmas=sigmas,
                seeds=seeds,
                experiment_config=experiment_config,
            )
            append_jsonl(handle, record)
            LOGGER.info(
                "id=%s answer=%r flip_auc=%.6f first_flip_sigma=%s seconds=%.2f",
                record["id"],
                record["original_answer"],
                record["uncertainty"]["flip_auc"],
                record["uncertainty"]["median_first_flip_sigma"],
                record["runtime_seconds"],
            )
    LOGGER.info("complete output=%s", args.output)


if __name__ == "__main__":
    main()
