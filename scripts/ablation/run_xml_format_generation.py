#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation.xml_format import DATASETS, dataset_sources, load_manifest, materialize_generation_subset, selected_ids
from src.generation.runner import run_generation
from src.models import load_generation_backend
from src.models.runtime import visible_gpu_memory_gib, vllm_max_model_len, vllm_max_num_seqs


MODEL_CONFIG = {
    "llava": ("llava_1_5", "llava-1.5-7b-hf"),
    "qwen": ("qwen2_5_vl", "Qwen2.5-VL-7B-Instruct"),
    "internvl": ("internvl3_5_original", "InternVL3_5-8B"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse XML-LoRA responses and generate matched native-model three-part responses."
    )
    parser.add_argument("--model", required=True, choices=tuple(MODEL_CONFIG))
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "results/ablation/xml_format/sample_manifest.json",
    )
    parser.add_argument(
        "--production-generation-root",
        type=Path,
        default=PROJECT_ROOT / "results/generation",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/ablation/xml_format/generation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=0)
    args = parser.parse_args()
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required on offline compute nodes")

    manifest = load_manifest(args.manifest)
    sources = dataset_sources(args.data_root)
    for dataset in DATASETS:
        count = materialize_generation_subset(
            source=args.production_generation_root / args.model / "greedy" / f"{dataset}.jsonl",
            output=args.output_root / "xml_lora" / args.model / f"{dataset}.jsonl",
            sample_ids=selected_ids(manifest, dataset),
            manifest=manifest,
        )
        print(f"materialized xml_lora {args.model}/{dataset}: {count}")

    family, model_dir = MODEL_CONFIG[args.model]
    model_path = args.model_root / model_dir
    memory_gib = visible_gpu_memory_gib()
    max_num_seqs = args.max_num_seqs or vllm_max_num_seqs(memory_gib)
    max_model_len = args.max_model_len or vllm_max_model_len(family)
    backend = load_generation_backend(
        family,
        model_path,
        adapter_path=None,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    for dataset in DATASETS:
        written, skipped = run_generation(
            backend=backend,
            family=family,
            model_path=model_path,
            dataset=dataset,
            dataset_source=sources[dataset],
            output=args.output_root / "native_prompt" / args.model / f"{dataset}.jsonl",
            max_new_tokens=args.max_new_tokens,
            num_samples=0,
            seed=manifest["seed"],
            limit=None,
            prompt_style="native_three_part",
            reject_resample_k=1,
            max_batch_size=max_num_seqs,
            request_window_samples=max(16, max_num_seqs * 2),
            phase="greedy",
            sample_ids=selected_ids(manifest, dataset),
        )
        print(f"generated native_prompt {args.model}/{dataset}: written={written} skipped={skipped}")


if __name__ == "__main__":
    main()
