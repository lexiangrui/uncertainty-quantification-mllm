#!/usr/bin/env python3
"""Generate COCO descriptions with pre/post-projector MALP perturbations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from perturb import PerturbSpec, perturb_tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/opt/lexiangrui/models/llava-1.5-7b-hf")
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--question", default="Describe this image briefly.")
    p.add_argument("--mode", choices=["norm_isotropic", "directional"], default="norm_isotropic")
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def make_inputs(processor, image: Image.Image, question: str, device: torch.device):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    return {name: value.to(device) for name, value in inputs.items()}


@torch.inference_mode()
def generate(model, processor, inputs, max_new_tokens: int) -> str:
    sequences = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.decode(sequences[0, prompt_len:], skip_special_tokens=True).strip()


def perturbed_generate(model, processor, inputs, stage: str, mode: str, sigma: float, seed: int, max_new_tokens: int) -> str:
    handles = []
    spec = PerturbSpec("vision", "fusion", mode, sigma, 1.0, seed)

    if stage == "pre_projector":
        # Projector input is the patch-token tensor produced by the vision tower.
        def projector_pre_hook(_module, args, kwargs):
            if kwargs:
                value = kwargs.get("hidden_states", args[0] if args else None)
                changed = perturb_tensor(value, spec)
                if "hidden_states" in kwargs:
                    kwargs["hidden_states"] = changed
                    return args, kwargs
                return (changed, *args[1:]), kwargs
            value = args[0]
            return (perturb_tensor(value, spec), *args[1:]), {}

        handles.append(model.model.multi_modal_projector.register_forward_pre_hook(projector_pre_hook, with_kwargs=True))
    elif stage == "post_projector":
        def projector_hook(_module, _args, output):
            return perturb_tensor(output, spec)

        handles.append(model.model.multi_modal_projector.register_forward_hook(projector_hook))
    else:
        raise ValueError(stage)
    try:
        return generate(model, processor, inputs, max_new_tokens)
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def main() -> None:
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg.model,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="cuda:0" if device.type == "cuda" else None,
        local_files_only=True,
        attn_implementation="sdpa",
    ).eval()
    if device.type == "cpu":
        model.to(device)
    image_paths = sorted(cfg.image_dir.glob("*.jpg"))[: cfg.max_images]
    if len(image_paths) != cfg.max_images:
        raise RuntimeError(f"expected {cfg.max_images} jpg files, found {len(image_paths)}")
    records = []
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        base_inputs = make_inputs(processor, image, cfg.question, device)
        original = generate(model, processor, base_inputs, cfg.max_new_tokens)
        pre = perturbed_generate(model, processor, base_inputs, "pre_projector", cfg.mode, cfg.sigma, cfg.seed + index * 10 + 1, cfg.max_new_tokens)
        post = perturbed_generate(model, processor, base_inputs, "post_projector", cfg.mode, cfg.sigma, cfg.seed + index * 10 + 2, cfg.max_new_tokens)
        records.append({
            "coco_image_id": path.stem,
            "image": str(path),
            "question": cfg.question,
            "original_description": original,
            "pre_projector_perturbed_description": pre,
            "post_projector_perturbed_description": post,
            "pre_projector_changed": pre != original,
            "post_projector_changed": post != original,
        })
        print(json.dumps(records[-1], ensure_ascii=False))
    payload = {
        "model": cfg.model,
        "dataset": "COCO val2017",
        "experiment": "prompt-based generation under pre/post projector visual perturbation",
        "configuration": {"mode": cfg.mode, "sigma": cfg.sigma, "seed": cfg.seed, "max_new_tokens": cfg.max_new_tokens},
        "records": records,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
