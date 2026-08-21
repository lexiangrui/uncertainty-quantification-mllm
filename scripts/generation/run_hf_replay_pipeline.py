#!/usr/bin/env python3
"""Replay all vLLM outputs while keeping one HF model resident."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generation.replay_hf_artifacts import replay_file
from src.models import load_replay_backend
from src.models.runtime import replay_batch_size, visible_gpu_memory_gib


def _sources(data_root: Path) -> dict[str, Path]:
    return {
        "vilp": data_root / "vilp/ViLP.parquet",
        "hallusionbench": data_root / "HallusionBench/data",
        "mmvet": data_root / "MMVet/data/test-00000-of-00001.parquet",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent HF replay pipeline")
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    memory_gib = visible_gpu_memory_gib()
    batch_size = args.batch_size or replay_batch_size(memory_gib)
    print(f"HF replay GPU memory={memory_gib:.1f} GiB batch_size={batch_size}")
    backend = load_replay_backend(
        args.model_family,
        args.model_path,
        adapter_path=args.adapter_path,
        attn_implementation=(
            "sdpa" if args.model_family == "internvl3_5_original" else "flash_attention_2"
        ),
    )
    raw_root = args.output_root / "vllm_raw"
    for dataset, source in _sources(args.data_root).items():
        for phase in ("greedy", "samples"):
            written, skipped = replay_file(
                input_path=raw_root / phase / f"{dataset}.jsonl",
                output_path=args.output_root / phase / f"{dataset}.jsonl",
                dataset_source=source,
                family=args.model_family,
                model_path=args.model_path,
                adapter_path=args.adapter_path,
                batch_size=batch_size,
                limit=args.limit,
                backend=backend,
            )
            print(f"HF replay {dataset}/{phase}: written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
