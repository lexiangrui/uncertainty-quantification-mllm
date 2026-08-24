#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.human_alignment.automated import run_automated_adjudication  # noqa: E402
from src.llm_judge import ClosedSourceJudge  # noqa: E402
from src.llm_judge.paths import judge_directory_name  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporarily adjudicate unresolved dual-judge disagreements with a third model."
    )
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "results/human_alignment")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()
    output = args.workspace / f"{judge_directory_name(args.model)}.jsonl"
    lock_path = args.workspace / ".automated_adjudication.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another automated adjudication process is already running") from error
        judge = ClosedSourceJudge(
            args.model,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout,
        )
        summary = run_automated_adjudication(
            judge=judge,
            workspace=args.workspace,
            output=output,
            concurrency=args.concurrency,
            limit=args.limit,
            checkpoint_every=args.checkpoint_every,
        )
    print(json.dumps({"model": args.model, "audit_output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
