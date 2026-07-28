#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_format.teacher import (  # noqa: E402
    PROMPT_VERSION,
    build_messages,
    create_teacher_client,
    request_teacher_payload,
)
from lora_format.validation import ValidationError, validate_teacher_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate grounded supervision with Qwen3.7-Plus.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def accepted_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {row["question_id"] for row in read_jsonl(path)}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def generate_one(client, model: str, row: dict, image_dir: Path, system_prompt: str, examples: list[dict]) -> tuple[str, dict]:
    audit = {
        "question_id": row["question_id"],
        "teacher_model": model,
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "few_shot_sha256": hashlib.sha256(
            json.dumps(examples, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
        "source_agreement": row["agreement"],
        "attempted_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        messages = build_messages(
            row,
            image_dir / row["image_file"],
            system_prompt,
            examples,
        )
        raw = request_teacher_payload(client, model, messages)
        clean = validate_teacher_payload(raw, row["answer"])
        return "accepted", {**row, "teacher": clean, **audit, "validation": "accepted"}
    except Exception as error:
        return "rejected", {
            **audit,
            "validation": "rejected",
            "reason": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    args = parse_args()
    # Validate credentials before reading images or creating output files.
    client = create_teacher_client()
    system_prompt = (ROOT / "prompts" / "teacher_prompt.md").read_text(encoding="utf-8").strip()
    examples = json.loads((ROOT / "prompts" / "few_shot_examples.json").read_text(encoding="utf-8"))
    done = accepted_ids(args.accepted)
    pending = [row for row in read_jsonl(args.candidates) if row["question_id"] not in done]
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        pending = pending[: args.limit]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                generate_one,
                client,
                args.model,
                row,
                args.image_dir,
                system_prompt,
                examples,
            ): row["question_id"]
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            status, output = future.result()
            append_jsonl(args.accepted if status == "accepted" else args.rejected, output)
            print(
                json.dumps(
                    {
                        "progress": index,
                        "total": len(pending),
                        "question_id": futures[future],
                        "status": status,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
