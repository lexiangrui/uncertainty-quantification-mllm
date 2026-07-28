import json
from pathlib import Path
from typing import TextIO


def batches(items: list, size: int):
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_jsonl_by_id(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"missing id at {path}:{line_number}")
            if record_id in records:
                raise ValueError(f"duplicate id {record_id!r} in {path}")
            records[record_id] = record
    return records


def append_jsonl(handle: TextIO, record: dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
