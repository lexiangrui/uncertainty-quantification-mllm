#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation.xml_format import (
    DATASETS,
    dataset_sources,
    load_manifest,
    materialize_existing_judge_subset,
    selected_ids,
)
from src.llm_judge import ClosedSourceJudge
from src.llm_judge.paths import judge_directory_name
from src.llm_judge.runner import run_closed_source_judging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse official XML labels and judge only native-prompt responses."
    )
    parser.add_argument("--tested-model", required=True, choices=("llava", "qwen", "internvl"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=PROJECT_ROOT / "results/ablation/xml_format/generation",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results/ablation/xml_format",
    )
    parser.add_argument(
        "--xml-judge-root",
        type=Path,
        default=PROJECT_ROOT / "results/judging_gemini_3_7_flash",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "results/ablation/xml_format/sample_manifest.json",
    )
    parser.add_argument("--judge-model", default="gemini-3.7-flash")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    sources = dataset_sources(args.data_root)
    judge_root = args.output_root / judge_directory_name(args.judge_model)
    for dataset in DATASETS:
        count = materialize_existing_judge_subset(
            source=args.xml_judge_root / args.tested_model / f"{dataset}.jsonl",
            output=judge_root / "xml_lora" / args.tested_model / f"{dataset}.jsonl",
            sample_ids=selected_ids(manifest, dataset),
            manifest=manifest,
            judge_model=args.judge_model,
        )
        print(f"reused Gemini XML labels {args.tested_model}/{dataset}: {count}")
    judge = ClosedSourceJudge(
        args.judge_model,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
    )
    for dataset in DATASETS:
        written, skipped = run_closed_source_judging(
            judge=judge,
            dataset=dataset,
            dataset_source=sources[dataset],
            greedy_input=args.generation_root / "native_prompt" / args.tested_model / f"{dataset}.jsonl",
            output=judge_root / "native_prompt" / args.tested_model / f"{dataset}.jsonl",
            limit=None,
            concurrency=args.concurrency,
            judge_raw_response=True,
        )
        print(
            f"judged native_prompt/{args.tested_model}/{dataset}: "
            f"written={written} skipped={skipped}"
        )


if __name__ == "__main__":
    main()
