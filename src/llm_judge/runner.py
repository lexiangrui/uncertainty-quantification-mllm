from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

_API_MAX_RETRIES = 3
_API_RETRY_BACKOFF_SECONDS = 15

from src.datasets import iter_dataset
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line

from .closed_source import ClosedSourceJudge, JUDGE_PROMPT_SHA256


def _load_greedy_records(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    run: dict[str, Any] | None = None
    records: dict[str, dict[str, Any]] = {}
    rows = load_jsonl_records(path)
    for index, record in enumerate(rows):
        if index == 0 and record.get("record_type") == "run":
            run = record.get("run")
            continue
        if run is None:
            run = record.get("run")
        elif "run" in record and record.get("run") != run:
            raise ValueError(f"greedy run mismatch at record {index}")
        sample_id = record.get("sample", {}).get("sample_id")
        if not isinstance(sample_id, str) or sample_id in records:
            raise ValueError(f"invalid or duplicate sample_id at record {index}")
        records[sample_id] = record
    if run is None:
        raise ValueError(f"greedy input is empty: {path}")
    return run, records


_INVALID_FORMAT_VALUE = {
    "status": "invalid_input",
    "valid": False,
    "error": "greedy response cannot be separated into three parts",
    "raw_response": None,
    "analysis": None,
    "correct": None,
    "rating": None,
    "hallucination": None,
    "hallucination_types": None,
}


def _judge_one(judge, sample, greedy: dict) -> dict:
    """Judge a single sample and retain its response-local raw audit text."""
    if greedy.get("sections_valid") is not True:
        return dict(_INVALID_FORMAT_VALUE)
    result = None
    api_error: Exception | None = None
    for _attempt in range(_API_MAX_RETRIES):
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
            api_error = None
            break
        except ValueError as error:
            api_error = error
            break
        except Exception as error:  # noqa: BLE001 — retry any API/transport error
            api_error = error
            time.sleep(_API_RETRY_BACKOFF_SECONDS)
    raw = result.raw_response if result is not None else getattr(api_error, "raw_response", None)
    if result is not None:
        return {"status": "ok", "valid": True, "error": None, **result.to_dict()}
    if isinstance(api_error, ValueError):
        return {
            "status": "invalid_response",
            "valid": False,
            "error": str(api_error),
            "raw_response": raw,
            "analysis": None,
            "correct": None,
            "rating": None,
            "hallucination": None,
            "hallucination_types": None,
        }
    return {
        "status": "api_error",
        "valid": False,
        "error": f"{type(api_error).__name__}: {api_error}",
        "raw_response": raw,
        "analysis": None,
        "correct": None,
        "rating": None,
        "hallucination": None,
        "hallucination_types": None,
    }


def run_closed_source_judging(
    *,
    judge: ClosedSourceJudge,
    dataset: str,
    dataset_source: Path,
    greedy_input: Path,
    output: Path,
    limit: int | None,
    concurrency: int = 1,
) -> tuple[int, int]:
    greedy_run, generation_records = _load_greedy_records(greedy_input)
    run = {
        "protocol": "openai-responses",
        "judge_model": judge.model,
        "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
        "max_tokens": judge.max_tokens,
        "dataset": dataset,
        "dataset_source": str(dataset_source.resolve()),
        "greedy_input": str(greedy_input.resolve()),
        "greedy_run": greedy_run,
    }
    completed = completed_sample_ids(output, run, retry_statuses={"api_error"})
    written = 0
    skipped = 0
    seen: set[str] = set()

    def _record(sample, greedy, judge_value: dict) -> None:
        nonlocal written
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

    batch_size = max(concurrency * 4, 4)

    def _flush(batch: list[tuple[Any, dict]]) -> None:
        if not batch:
            return
        if concurrency <= 1 or len(batch) == 1:
            for sample, greedy in batch:
                _record(sample, greedy, _judge_one(judge, sample, greedy))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(_judge_one, judge, sample, greedy): (sample, greedy)
                    for sample, greedy in batch
                }
                for future in as_completed(futures):
                    sample, greedy = futures[future]
                    _record(sample, greedy, future.result())
        batch.clear()

    batch: list[tuple[Any, dict]] = []
    for sample in iter_dataset(dataset, dataset_source):
        record = generation_records.get(sample.sample_id)
        if record is None:
            continue
        seen.add(sample.sample_id)
        if limit is not None and written + skipped + len(batch) >= limit:
            break
        if sample.sample_id in completed:
            skipped += 1
            continue
        batch.append((sample, record.get("greedy", {})))
        if len(batch) >= batch_size:
            _flush(batch)
    _flush(batch)

    missing = set(generation_records) - seen
    if limit is None and missing:
        raise ValueError(f"generation records not found in dataset: {sorted(missing)[:3]}")
    return written, skipped
