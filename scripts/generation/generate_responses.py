#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.generation.runner import run_generation
from src.models import load_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one greedy response and shared sampled responses offline."
    )
    parser.add_argument("--dataset", required=True, choices=("vilp", "hallusionbench", "mmvet"))
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument(
        "--model-family",
        required=True,
        choices=("llava_1_5", "qwen2_5_vl", "internvl3_5"),
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reject-resample-k", type=int, default=10)
    parser.add_argument("--greedy-reject-resample-k", type=int, default=10)
    parser.add_argument("--greedy-recovery-temperature", type=float, default=0.2)
    parser.add_argument("--sampling-batch-size", type=int, default=5)
    parser.add_argument(
        "--prompt-style",
        choices=("xml_lora",),
        default="xml_lora",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--attn-implementation")
    return parser.parse_args()


def require_offline_mode() -> None:
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    missing = [name for name in required if os.environ.get(name) != "1"]
    if missing:
        raise RuntimeError(f"offline generation requires environment variables set to 1: {missing}")


def main() -> None:
    args = parse_args()
    require_offline_mode()
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.num_samples <= 0:
        raise ValueError("num-samples must be positive")
    if args.reject_resample_k <= 0:
        raise ValueError("reject-resample-k must be positive")
    if args.greedy_reject_resample_k <= 0:
        raise ValueError("greedy-reject-resample-k must be positive")
    if args.greedy_recovery_temperature <= 0:
        raise ValueError("greedy-recovery-temperature must be positive")
    if args.sampling_batch_size <= 0:
        raise ValueError("sampling-batch-size must be positive")
    backend = load_backend(
        args.model_family,
        args.model_path,
        attn_implementation=args.attn_implementation,
        adapter_path=args.adapter_path,
    )
    written, skipped = run_generation(
        backend=backend,
        family=args.model_family,
        model_path=args.model_path,
        dataset=args.dataset,
        dataset_source=args.dataset_source,
        output=args.output,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
        seed=args.seed,
        limit=args.limit,
        uq_methods=(),
        prompt_style=args.prompt_style,
        reject_resample_k=args.reject_resample_k,
        greedy_reject_resample_k=args.greedy_reject_resample_k,
        greedy_recovery_temperature=args.greedy_recovery_temperature,
        sampling_batch_size=args.sampling_batch_size,
    )
    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
