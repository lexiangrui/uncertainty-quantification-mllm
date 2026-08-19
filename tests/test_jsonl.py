from __future__ import annotations

import pytest

from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line


def test_completed_ids_allows_retryable_error_before_success(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    run = {"name": "test"}
    write_sample_json_line(path, run, {"sample": {"sample_id": "failed"}, "status": "error"})
    write_sample_json_line(path, run, {"sample": {"sample_id": "done"}, "status": "ok"})

    completed = completed_sample_ids(path, run, retry_statuses={"error"})

    assert completed == {"done"}


def test_load_jsonl_records_rejects_truncated_multiline_json(tmp_path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"ok": 1}\n{"broken":\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_jsonl_records(path)
