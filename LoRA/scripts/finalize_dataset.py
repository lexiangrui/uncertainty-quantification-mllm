#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_format.validation import validate_teacher_payload  # noqa: E402
from lora_format.xml import build_xml_response  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge teacher JSONL files into 4:1 XML SFT splits.")
    parser.add_argument("--accepted", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_rows = [row for path in args.accepted for row in read_jsonl(path)]
    expected = args.train_size + args.validation_size
    if len(source_rows) != expected:
        raise RuntimeError(f"expected {expected} source records, found {len(source_rows)}")
    question_ids = [row["question_id"] for row in source_rows]
    image_ids = [row["image_id"] for row in source_rows]
    if len(set(question_ids)) != expected:
        raise RuntimeError("question_id values must be unique across all teacher files")
    if len(set(image_ids)) != expected:
        raise RuntimeError("image_id values must be unique across all teacher files")

    rows = []
    for row in source_rows:
        clean = validate_teacher_payload(row["teacher"], row["answer"])
        rows.append(
            {
                **{key: value for key, value in row.items() if key not in {"teacher", "split"}},
                "response": build_xml_response(**clean),
            }
        )
    random.Random(args.seed).shuffle(rows)
    train = [{**row, "split": "train"} for row in rows[: args.train_size]]
    validation = [{**row, "split": "validation"} for row in rows[args.train_size :]]
    if {row["image_id"] for row in train} & {row["image_id"] for row in validation}:
        raise RuntimeError("image leakage between train and validation")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(args.output_dir / "train.jsonl", train)
    write_atomic(args.output_dir / "validation.jsonl", validation)
    print(
        json.dumps(
            {
                "sample_count": expected,
                "train": len(train),
                "validation": len(validation),
                "test": 0,
                "seed": args.seed,
            }
        )
    )


if __name__ == "__main__":
    main()
