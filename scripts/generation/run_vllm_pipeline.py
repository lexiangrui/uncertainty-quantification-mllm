#!/usr/bin/env python3
"""Run all vLLM generation phases while keeping one engine resident."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.runner import run_generation
from src.models import load_generation_backend
from src.models.runtime import visible_gpu_memory_gib, vllm_max_num_seqs


def _sources(data_root: Path) -> dict[str, Path]:
    return {
        "vilp": data_root / "vilp/ViLP.parquet",
        "hallusionbench": data_root / "HallusionBench/data",
        "mmvet": data_root / "MMVet/data/test-00000-of-00001.parquet",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent vLLM generation pipeline")
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required on offline compute nodes")

    memory_gib = visible_gpu_memory_gib()
    max_num_seqs = args.max_num_seqs or vllm_max_num_seqs(memory_gib)
    print(f"vLLM GPU memory={memory_gib:.1f} GiB max_num_seqs={max_num_seqs}")
    backend = load_generation_backend(
        args.model_family,
        args.model_path,
        adapter_path=args.adapter_path,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    raw_root = args.output_root / "vllm_raw"
    for dataset, source in _sources(args.data_root).items():
        for phase, num_samples in (("greedy", 0), ("samples", 10)):
            output = raw_root / phase / f"{dataset}.jsonl"
            written, skipped = run_generation(
                backend=backend,
                family=args.model_family,
                model_path=args.model_path,
                dataset=dataset,
                dataset_source=source,
                output=output,
                max_new_tokens=args.max_new_tokens,
                num_samples=num_samples,
                seed=42,
                limit=args.limit,
                prompt_style="xml_lora",
                reject_resample_k=50,
                max_batch_size=max_num_seqs,
                request_window_samples=max(16, max_num_seqs * 2),
                phase=phase,
            )
            print(f"vLLM {dataset}/{phase}: written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
