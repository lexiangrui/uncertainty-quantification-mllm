from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.datasets import iter_dataset
from src.utils import completed_sample_ids, write_sample_json_line

from .openai_chat import JUDGE_PROMPT_VERSION, OpenAIChatJudge


def _load_generation_records(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    generation_run: dict[str, Any] | None = None
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if line_number == 1 and record.get("record_type") == "run":
                generation_run = record.get("run")
                continue
            if generation_run is None:
                generation_run = record.get("run")
            elif "run" in record and record.get("run") != generation_run:
                raise ValueError(f"generation run mismatch at {path}:{line_number}")
            sample_id = record.get("sample", {}).get("sample_id")
            if not isinstance(sample_id, str) or sample_id in records:
                raise ValueError(f"invalid or duplicate sample_id at {path}:{line_number}")
            records[sample_id] = record
    if generation_run is None:
        raise ValueError(f"generation input is empty: {path}")
    return generation_run, records


def run_openai_judging(
    *,
    judge: OpenAIChatJudge,
    dataset: str,
    dataset_source: Path,
    generation_input: Path,
    output: Path,
    limit: int | None,
) -> tuple[int, int]:
    generation_run, generation_records = _load_generation_records(generation_input)
    run = {
        "protocol": "openai-chat-completions",
        "judge_model": judge.model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "max_tokens": judge.max_tokens,
        "dataset": dataset,
        "dataset_source": str(dataset_source.resolve()),
        "generation_input": str(generation_input.resolve()),
        "generation_run": generation_run,
    }
    completed = completed_sample_ids(output, run)
    written = 0
    skipped = 0
    seen: set[str] = set()
    matched = 0
    for sample in iter_dataset(dataset, dataset_source):
        record = generation_records.get(sample.sample_id)
        if record is None:
            continue
        if limit is not None and matched >= limit:
            break
        matched += 1
        seen.add(sample.sample_id)
        if sample.sample_id in completed:
            skipped += 1
            continue
        greedy = record.get("greedy", {})
        if greedy.get("sections_valid") is not True:
            judge_value = {
                "valid": False,
                "error": "greedy response cannot be separated into three parts",
                "raw_response": None,
                "analysis": None,
                "correct": None,
                "rating": None,
                "hallucination": None,
                "hallucination_types": None,
            }
        else:
            try:
                result = judge.judge(
                    dataset=sample.dataset,
                    question=sample.question,
                    references=list(sample.references),
                    vision=greedy["vision"],
                    reasoning=greedy["reasoning"],
                    answer=greedy["answer"],
                    image=sample.image,
                )
            except ValueError as error:
                judge_value = {
                    "valid": False,
                    "error": str(error),
                    "raw_response": judge.last_raw_response,
                    "analysis": None,
                    "correct": None,
                    "rating": None,
                    "hallucination": None,
                    "hallucination_types": None,
                }
            else:
                judge_value = {
                    "valid": True,
                    "error": None,
                    "raw_response": judge.last_raw_response,
                    **result.to_dict(),
                }
        write_sample_json_line(
            output,
            run,
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
                    "vision": greedy["vision"],
                    "reasoning": greedy["reasoning"],
                    "answer": greedy["answer"],
                    "raw_response": greedy["raw_response"],
                },
                "judge": judge_value,
            },
        )
        written += 1
    missing = set(generation_records) - seen
    if limit is None and missing:
        raise ValueError(f"generation records not found in dataset: {sorted(missing)[:3]}")
    return written, skipped
