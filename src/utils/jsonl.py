from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from a file, handling both compact and multi-line JSON.

    Tries per-line parsing first (fast path for compact JSONL).  If that
    fails, falls back to a streaming decoder that handles pretty-printed
    multi-line JSON objects.
    """
    with path.open(encoding="utf-8") as handle:
        content = handle.read()

    # Fast path: each non-empty line is a complete JSON object
    lines = [line for line in content.splitlines() if line.strip()]
    try:
        return [json.loads(line) for line in lines]
    except json.JSONDecodeError:
        pass

    # Fallback: streaming decode for multi-line JSON
    decoder = json.JSONDecoder()
    results: list[dict[str, Any]] = []
    idx = 0
    while idx < len(content):
        s = content[idx:].lstrip()
        if not s:
            break
        offset = len(content) - len(s)
        try:
            obj, end = decoder.raw_decode(s)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON while parsing {path} near character {idx + offset}"
            ) from error
        idx = offset + end
        if isinstance(obj, dict):
            results.append(obj)
    return results


_EXECUTION_ONLY_TOP_LEVEL_KEYS = frozenset({"replay_batch_size"})
_EXECUTION_ONLY_NESTED_KEYS = {
    "scheduler": frozenset(
        {
            "max_batch_size",
            "request_window_samples",
            "max_num_seqs",
            "adaptive_oom_split",
        }
    ),
    "model_runtime": frozenset(
        {"max_num_seqs", "gpu_memory_utilization"}
    ),
}


def _result_defining_run_config(run: Any) -> Any:
    """Remove throughput-only knobs while retaining all result-defining fields."""
    if not isinstance(run, dict):
        return run
    normalized = {
        key: value
        for key, value in run.items()
        if key not in _EXECUTION_ONLY_TOP_LEVEL_KEYS
    }
    for key, ignored_keys in _EXECUTION_ONLY_NESTED_KEYS.items():
        section = normalized.get(key)
        if isinstance(section, dict):
            ignored = ignored_keys
            if (
                key == "model_runtime"
                and normalized.get("internal_state_engine")
                == "hf_teacher_forcing"
            ):
                # Replay consumes frozen token IDs. The upstream vLLM context
                # capacity cannot change replay probabilities or hidden states.
                ignored = ignored | {"max_model_len"}
            normalized[key] = {
                name: value
                for name, value in section.items()
                if name not in ignored
            }
    return normalized


def completed_sample_ids(
    path: Path,
    run: dict[str, Any],
    *,
    retry_statuses: set[str] | None = None,
) -> set[str]:
    """Return completed ids, optionally allowing specified failed statuses to retry.

    Retried records remain in the JSONL audit trail. A later successful record
    for the same sample is therefore valid when earlier records have one of
    ``retry_statuses``; duplicate completed records remain an error.
    """
    if not path.exists():
        return set()
    completed: set[str] = set()
    layout: str | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if line_number == 1 and record.get("record_type") == "run":
                if _result_defining_run_config(
                    record.get("run")
                ) != _result_defining_run_config(run):
                    raise ValueError(
                        f"run configuration mismatch at {path}:{line_number}"
                    )
                layout = "header"
                continue
            if layout is None:
                layout = "inline"
            if layout == "header":
                if record.get("record_type") != "sample" or "run" in record:
                    raise ValueError(f"invalid sample record at {path}:{line_number}")
            elif _result_defining_run_config(
                record.get("run")
            ) != _result_defining_run_config(run):
                raise ValueError(f"run configuration mismatch at {path}:{line_number}")
            sample_id = record.get("sample", {}).get("sample_id")
            if not isinstance(sample_id, str):
                raise ValueError(f"missing sample_id at {path}:{line_number}")
            if retry_statuses is not None and record.get("status") in retry_statuses:
                continue
            if sample_id in completed:
                raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
            completed.add(sample_id)
    return completed


def write_sample_json_line(
    path: Path, run: dict[str, Any], record: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        if first.get("record_type") == "run":
            payload = {"record_type": "sample", **record}
        else:
            payload = {"run": run, **record}
        write_json_line(path, payload)
        return
    header = json.dumps(
        {"record_type": "run", "run": run},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    sample = json.dumps(
        {"record_type": "sample", **record},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(header + "\n" + sample + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json_line(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())
