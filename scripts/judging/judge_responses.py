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
    parser.add_argument("--greedy-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        default="gpt-5.6-terra",
        help="Judge model (default: gpt-5.6-terra)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request API timeout in seconds (default: 300)",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", "--max-workers", dest="concurrency", type=int, default=10,
                        help="Number of concurrent judge API requests (default: 10)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge = ClosedSourceJudge(
        args.model, max_tokens=args.max_tokens, timeout_seconds=args.timeout
    )
    written, skipped = run_closed_source_judging(
        judge=judge,
        dataset=args.dataset,
        dataset_source=args.dataset_source,
        greedy_input=args.greedy_input,
        output=args.output,
        limit=args.limit,
        concurrency=args.concurrency,
    )
    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
