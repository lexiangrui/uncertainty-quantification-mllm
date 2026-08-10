#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_judge import ClosedSourceJudge
from src.llm_judge.runner import run_closed_source_judging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge generated multimodal responses through the OpenAI Responses API."
    )
    parser.add_argument("--dataset", required=True, choices=("vilp", "hallusionbench", "mmvet"))
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument("--generation-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge = ClosedSourceJudge(args.model, max_tokens=args.max_tokens)
    written, skipped = run_closed_source_judging(
        judge=judge,
        dataset=args.dataset,
        dataset_source=args.dataset_source,
        generation_input=args.generation_input,
        output=args.output,
        limit=args.limit,
        concurrency=args.concurrency,
    )
    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
