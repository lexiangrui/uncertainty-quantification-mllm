#!/usr/bin/env python3
"""Perturb five high-NLL wrong ViLP answers before/after the projector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import iter_samples  # noqa: E402
from model import LlavaMalpRunner  # noqa: E402
from perturb import PerturbSpec, perturb_tensor  # noqa: E402


# Selected from the 50-sample baseline as correct predictions with the highest
# mean NLL (strict normalized prediction/reference matching).
WRONG_TARGET_IDS = ["vilp-17-case2", "vilp-3-case2", "vilp-2-case2", "vilp-0-case2", "vilp-1-case1"]
CORRECT_TARGET_IDS = ["vilp-2-case1", "vilp-14-case1", "vilp-12-case2", "vilp-1-case2", "vilp-0-case1"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--mode", choices=["norm_isotropic", "directional"], default="norm_isotropic")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--group", choices=["wrong", "correct"], default="wrong")
    return p.parse_args()


def nll_mask(runner: LlavaMalpRunner, answer_ids: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    mask = answer_mask.bool().clone()
    for eos_id in runner.eos_token_ids:
        mask &= answer_ids.ne(eos_id)
    for pos in range(answer_ids.shape[1]):
        token_id = int(answer_ids[0, pos].item())
        text = runner.processor.tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not text.strip():
            mask[:, pos] = False
    return mask


def install_projector_hook(runner: LlavaMalpRunner, stage: str, spec: PerturbSpec) -> list:
    if stage == "pre_projector":
        def pre_hook(_module, args, kwargs):
            value = kwargs.get("hidden_states", args[0] if args else None)
            changed = perturb_tensor(value, spec)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = changed
                return args, kwargs
            return (changed, *args[1:]), kwargs
        return [runner.projector.register_forward_pre_hook(pre_hook, with_kwargs=True)]
    if stage == "post_projector":
        def post_hook(_module, _args, output):
            return perturb_tensor(output, spec)
        return [runner.projector.register_forward_hook(post_hook)]
    raise ValueError(stage)


@torch.inference_mode()
def generate(runner: LlavaMalpRunner, inputs: dict, max_new_tokens: int) -> tuple[str, torch.Tensor, torch.Tensor]:
    prompt_len = inputs["input_ids"].shape[1]
    seq = runner.model.generate(**runner._model_inputs(inputs), do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    ids = seq[:, prompt_len:]
    mask = runner.build_answer_mask(ids)
    return runner.processor.batch_decode(ids, skip_special_tokens=True)[0].strip(), ids, mask


def generate_perturbed(runner, inputs, stage, mode, sigma, seed, max_new_tokens):
    spec = PerturbSpec("vision", "fusion", mode, sigma, 1.0, seed)
    handles = install_projector_hook(runner, stage, spec)
    try:
        return generate(runner, inputs, max_new_tokens)
    finally:
        for h in handles:
            h.remove()


def teacher_nll(runner, inputs, answer_ids, answer_mask, stage=None, mode="norm_isotropic", sigma=0.1, seed=0):
    teacher = runner.build_teacher_forcing_inputs(inputs, answer_ids, answer_mask)
    handles = []
    if stage is not None:
        handles = install_projector_hook(runner, stage, PerturbSpec("vision", "fusion", mode, sigma, 1.0, seed))
    try:
        logits = runner.forward_original(teacher)["response_logits"]
    finally:
        for h in handles:
            h.remove()
    mask = nll_mask(runner, answer_ids, answer_mask)
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), answer_ids.reshape(-1), reduction="none").reshape(answer_ids.shape)
    return float(losses[mask].sum() / mask.sum().clamp_min(1)), int(mask.sum())


def main() -> None:
    cfg = parse_args()
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    runner = LlavaMalpRunner()
    target_ids = WRONG_TARGET_IDS if cfg.group == "wrong" else CORRECT_TARGET_IDS
    samples = {s["id"]: s for s in iter_samples("vilp", limit=50)}
    missing = [x for x in target_ids if x not in samples]
    if missing:
        raise RuntimeError(f"missing target samples: {missing}")
    records = []
    for index, sample_id in enumerate(target_ids):
        sample = samples[sample_id]
        inputs = runner.prepare_inputs(sample["image"], sample["question"])
        original_text, answer_ids, answer_mask = generate(runner, inputs, cfg.max_new_tokens)
        original_nll, nll_tokens = teacher_nll(runner, inputs, answer_ids, answer_mask)
        row = {
            "id": sample_id,
            "question": sample["question"],
            "reference": sample["references"],
            "original_prediction": original_text,
            "original_mean_nll": original_nll,
            "nll_token_count": nll_tokens,
            "perturbation": {},
        }
        for offset, stage in enumerate(("pre_projector", "post_projector")):
            text, _, _ = generate_perturbed(runner, inputs, stage, cfg.mode, cfg.sigma, 1000 + index * 10 + offset, cfg.max_new_tokens)
            pert_nll, _ = teacher_nll(runner, inputs, answer_ids, answer_mask, stage, cfg.mode, cfg.sigma, 2000 + index * 10 + offset)
            row["perturbation"][stage] = {
                "description": text,
                "description_changed": text != original_text,
                "teacher_forced_original_answer_nll": pert_nll,
                "delta_nll": pert_nll - original_nll,
                "seed_generation": 1000 + index * 10 + offset,
                "seed_teacher_forcing": 2000 + index * 10 + offset,
            }
        records.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {
        "dataset": "vilp",
        "selected_ids": target_ids,
        "group": cfg.group,
        "selection": f"{cfg.group} predictions ranked by descending original mean NLL",
        "mode": cfg.mode,
        "sigma": cfg.sigma,
        "records": records,
        "aggregate": {
            stage: {
                "description_changed_count": sum(r["perturbation"][stage]["description_changed"] for r in records),
                "mean_perturbed_nll": sum(r["perturbation"][stage]["teacher_forced_original_answer_nll"] for r in records) / len(records),
                "mean_delta_nll": sum(r["perturbation"][stage]["delta_nll"] for r in records) / len(records),
            }
            for stage in ("pre_projector", "post_projector")
        },
    }
    cfg.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
