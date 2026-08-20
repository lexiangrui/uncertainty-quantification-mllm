#!/usr/bin/env python3
"""Deterministic reference-match labels for smoke evaluation.

This is intentionally not a replacement for the production visual judge.  It
provides auditable labels for the requested 10-sample validation run without
network access: the answer is normalized and compared with the benchmark's
reference answers.  The run header records the exact rule version.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import iter_dataset
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line


RULE_VERSION = "local-reference-rule-v1"


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).lower().strip()
    value = re.sub(r"^[\s\[\](){}<>]*(?:answer|option|choice)\s*(?:is|:)?\s*", "", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _matches(answer: str, references: list[str]) -> bool:
    candidate = _normalize(answer)
    if not candidate:
        return False
    for reference in references:
        expected = _normalize(reference)
        if not expected:
            continue
        if candidate == expected or expected in candidate or candidate in expected:
            return True
    return False


def _judge_record(sample, greedy: dict) -> dict:
    if greedy.get("sections_valid") is not True:
        return {
            "status": "invalid_input",
            "valid": False,
            "error": "greedy response cannot be separated into three parts",
            "raw_response": greedy.get("raw_response"),
            "analysis": None,
            "correct": None,
            "rating": None,
            "hallucination": None,
            "hallucination_types": None,
        }
    correct = _matches(greedy.get("answer", ""), list(sample.references))
    hallucination = not correct
    return {
        "status": "ok",
        "valid": True,
        "error": None,
        "raw_response": "local-reference-rule-v1",
        "analysis": (
            "normalized answer matched at least one reference"
            if correct
            else "normalized answer did not match any reference"
        ),
        "correct": correct,
        "rating": 6 if correct else 0,
        "hallucination": hallucination,
        "hallucination_types": ["reasoning_hallucination"] if hallucination else [],
    }


def run(*, dataset: str, dataset_source: Path, greedy_input: Path, output: Path, limit: int | None) -> tuple[int, int]:
    rows = load_jsonl_records(greedy_input)
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError(f"greedy input lacks run header: {greedy_input}")
    greedy_run = dict(rows[0]["run"])
    generation = {
        row.get("sample", {}).get("sample_id"): row
        for row in rows[1:]
        if isinstance(row.get("sample", {}).get("sample_id"), str)
    }
    judge_run = {
        "protocol": RULE_VERSION,
        "judge_model": RULE_VERSION,
        "judge_rule_version": RULE_VERSION,
        "dataset": dataset,
        "dataset_source": str(dataset_source.resolve()),
        "greedy_input": str(greedy_input.resolve()),
        "greedy_run": greedy_run,
    }
    completed = completed_sample_ids(output, judge_run)
    written = skipped = 0
    seen: set[str] = set()
    for sample in iter_dataset(dataset, dataset_source, limit):
        record = generation.get(sample.sample_id)
        if record is None:
            skipped += 1
            continue
        seen.add(sample.sample_id)
        if sample.sample_id in completed:
            skipped += 1
            continue
        greedy = record.get("greedy", {})
        write_sample_json_line(
            output,
            judge_run,
            {
                "sample": {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "dataset": sample.dataset,
                    "split": sample.split,
                },
                "input": {
                    "question": sample.question,
                    "references": list(sample.references),
                    "vision": greedy.get("vision"),
                    "reasoning": greedy.get("reasoning"),
                    "answer": greedy.get("answer"),
                    "raw_response": greedy.get("raw_response"),
                },
                "judge": _judge_record(sample, greedy),
            },
        )
        written += 1
    if limit is None:
        missing = set(generation) - seen
        if missing:
            raise ValueError(f"generation records not found in dataset: {sorted(missing)[:3]}")
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply deterministic reference-match labels")
    parser.add_argument("--dataset", required=True, choices=("vilp", "hallusionbench", "mmvet"))
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument("--greedy-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    written, skipped = run(
        dataset=args.dataset,
        dataset_source=args.dataset_source,
        greedy_input=args.greedy_input,
        output=args.output,
        limit=args.limit,
    )
    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
