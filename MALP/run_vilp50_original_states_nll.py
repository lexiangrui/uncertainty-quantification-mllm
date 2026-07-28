#!/usr/bin/env python3
"""Run unperturbed LLaVA on 50 ViLP samples and record states/NLL."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import iter_samples  # noqa: E402
from model import LlavaMalpRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    return parser.parse_args()


def tensor_summary(value: torch.Tensor) -> dict:
    x = value.detach().float()
    token_norms = x.norm(dim=-1)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "frobenius_norm": float(x.norm()),
        "mean_token_norm": float(token_norms.mean()),
        "std_token_norm": float(token_norms.std()),
    }


@torch.inference_mode()
def generate_and_capture(
    runner: LlavaMalpRunner,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}

    def projector_pre_hook(_module, args):
        if "pre" not in captured:
            captured["pre"] = args[0].detach().to(dtype=torch.float16, device="cpu")

    def projector_hook(_module, _args, output):
        if "post" not in captured:
            captured["post"] = output.detach().to(dtype=torch.float16, device="cpu")

    handles = [
        runner.projector.register_forward_pre_hook(projector_pre_hook),
        runner.projector.register_forward_hook(projector_hook),
    ]
    try:
        prompt_length = inputs["input_ids"].shape[1]
        sequences = runner.model.generate(
            **runner._model_inputs(inputs),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"pre", "post"}:
        raise RuntimeError(f"failed to capture projector states: {sorted(captured)}")
    answer_ids = sequences[:, prompt_length:]
    if answer_ids.numel() == 0:
        raise RuntimeError("LLaVA generated an empty answer")
    answer_mask = runner.build_answer_mask(answer_ids)
    answer = runner.processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
    hit_length_limit = answer_ids.shape[1] >= max_new_tokens and not any(
        answer_ids[0, -1].item() == token_id for token_id in runner.eos_token_ids
    )
    return (
        {
            "text": answer,
            "answer_ids": answer_ids.detach(),
            "answer_mask": answer_mask.detach(),
            "generated_token_count": int(answer_mask.sum().item()),
            "hit_length_limit": bool(hit_length_limit),
        },
        captured["pre"],
        captured["post"],
    )


def main() -> None:
    args = parse_args()
    if args.limit != 50:
        raise ValueError("this experiment is fixed to exactly 50 ViLP samples")
    for path in (args.output, args.summary):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    runner = LlavaMalpRunner()
    sample_nlls: list[float] = []
    total_nll_sum = 0.0
    total_token_count = 0
    truncated_count = 0
    started = time.perf_counter()

    with args.output.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(iter_samples("vilp", limit=args.limit), start=1):
            sample_started = time.perf_counter()
            inputs = runner.prepare_inputs(sample["image"], sample["question"])
            generated, pre_state, post_state = generate_and_capture(
                runner, inputs, args.max_new_tokens
            )
            teacher = runner.build_teacher_forcing_inputs(
                inputs, generated["answer_ids"], generated["answer_mask"]
            )
            logits = runner.forward_original(teacher)["response_logits"]
            answer_ids = teacher["answer_ids"]
            answer_mask = teacher["answer_mask"]
            # Score the generated answer content, excluding the EOS terminator.
            # The previous version included EOS because build_answer_mask keeps
            # it for causal termination scoring; that is not the paper-style
            # answer-token NLL used by the official likelihood implementations.
            nll_mask = answer_mask.clone()
            for eos_id in runner.eos_token_ids:
                nll_mask &= answer_ids.ne(eos_id)
            # A standalone SentencePiece whitespace token (often rendered as
            # ``▁``) is tokenizer scaffolding, not visible answer content.
            # Exclude it from the paper-style answer-token NLL as well.
            for token_id in range(answer_ids.shape[1]):
                token_value = int(answer_ids[0, token_id].item())
                decoded_token = runner.processor.tokenizer.decode(
                    [token_value],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if not decoded_token.strip():
                    nll_mask[:, token_id] = False
            per_token_nll = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                answer_ids.reshape(-1),
                reduction="none",
            ).reshape(answer_ids.shape)
            valid = nll_mask.bool()
            nll_sum = float(per_token_nll[valid].sum().item())
            token_count = int(valid.sum().item())
            mean_nll = nll_sum / token_count
            sample_nlls.append(mean_nll)
            total_nll_sum += nll_sum
            total_token_count += token_count
            truncated_count += int(generated["hit_length_limit"])

            state_path = args.state_dir / f"{sample['id']}.pt"
            torch.save(
                {
                    "id": sample["id"],
                    "pre_projector": pre_state,
                    "post_projector": post_state,
                },
                state_path,
            )
            record = {
                "index": index,
                "id": sample["id"],
                "dataset": "vilp",
                "question": sample["question"],
                "references": sample["references"],
                "prediction": generated["text"],
                "generated_token_count": generated["generated_token_count"],
                "nll_token_count": token_count,
                "hit_length_limit": generated["hit_length_limit"],
                "mean_nll": mean_nll,
                "nll_sum": nll_sum,
                "state_file": str(state_path),
                "pre_projector_state": tensor_summary(pre_state),
                "post_projector_state": tensor_summary(post_state),
                "metadata": sample["metadata"],
                "runtime_seconds": time.perf_counter() - sample_started,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index:02d}/{args.limit}] {sample['id']} "
                f"tokens={token_count} nll={mean_nll:.6f} "
                f"pre={tuple(pre_state.shape)} post={tuple(post_state.shape)}",
                flush=True,
            )

    macro_mean_nll = sum(sample_nlls) / len(sample_nlls)
    summary = {
        "model": str(runner.model.config.name_or_path),
        "dataset": "vilp",
        "sample_count": len(sample_nlls),
        "max_new_tokens": args.max_new_tokens,
        "perturbation": None,
        "macro_mean_nll": macro_mean_nll,
        "token_weighted_mean_nll": total_nll_sum / total_token_count,
        "total_answer_tokens": total_token_count,
        "hit_length_limit_count": truncated_count,
        "state_storage": "full fp16 pre/post-projector tensors, one .pt file per sample",
        "state_dir": str(args.state_dir),
        "records": str(args.output),
        "runtime_seconds": time.perf_counter() - started,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
