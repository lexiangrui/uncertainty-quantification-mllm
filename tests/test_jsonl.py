from __future__ import annotations

from src.utils import completed_sample_ids, write_sample_json_line


def test_completed_ids_allows_retryable_error_before_success(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    run = {"name": "test"}
    write_sample_json_line(path, run, {"sample": {"sample_id": "failed"}, "status": "error"})
    write_sample_json_line(path, run, {"sample": {"sample_id": "done"}, "status": "ok"})

    completed = completed_sample_ids(path, run, retry_statuses={"error"})

    assert completed == {"done"}
