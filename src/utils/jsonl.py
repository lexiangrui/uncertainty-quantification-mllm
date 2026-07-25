from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def completed_sample_ids(path: Path, run: dict[str, Any]) -> set[str]:
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
                if record.get("run") != run:
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
            elif record.get("run") != run:
                raise ValueError(f"run configuration mismatch at {path}:{line_number}")
            sample_id = record.get("sample", {}).get("sample_id")
            if not isinstance(sample_id, str):
                raise ValueError(f"missing sample_id at {path}:{line_number}")
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
