#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hidden import extract_hidden_states


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute sampled-response hidden states after generation."
    )
    parser.add_argument("--generation-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model-family",
        required=True,
        choices=("llava_1_5", "qwen2_5_vl", "internvl3_5"),
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    args = parser.parse_args()
    required = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    missing = [name for name in required if os.environ.get(name) != "1"]
    if missing:
        raise RuntimeError(f"offline hidden extraction requires {missing} set to 1")
    written, skipped = extract_hidden_states(
        generation_input=args.generation_input,
        output=args.output,
        family=args.model_family,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        dataset_source=args.dataset_source,
        attn_implementation=args.attn_implementation,
    )
    print(
        f"completed hidden extraction: written={written} skipped={skipped} output={args.output}"
    )


if __name__ == "__main__":
    main()
