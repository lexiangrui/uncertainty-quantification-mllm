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


def test_completed_ids_ignore_throughput_only_runtime_changes(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    previous = {
        "name": "test",
        "replay_batch_size": 5,
        "scheduler": {
            "type": "vllm_role_separated_dynamic_batching",
            "max_batch_size": 5,
            "request_window_samples": 16,
        },
        "model_runtime": {
            "engine": "vllm",
            "max_model_len": 18_000,
            "max_num_seqs": 8,
            "gpu_memory_utilization": 0.85,
        },
    }
    current = {
        **previous,
        "replay_batch_size": 4,
        "scheduler": {
            **previous["scheduler"],
            "max_batch_size": 4,
            "request_window_samples": 8,
        },
        "model_runtime": {
            **previous["model_runtime"],
            "max_num_seqs": 6,
            "gpu_memory_utilization": 0.8,
        },
    }
    write_sample_json_line(path, previous, {"sample": {"sample_id": "done"}})

    assert completed_sample_ids(path, current) == {"done"}


def test_completed_ids_reject_max_model_len_change(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    previous = {
        "name": "test",
        "model_runtime": {"engine": "vllm", "max_model_len": 4096},
    }
    current = {
        "name": "test",
        "model_runtime": {"engine": "vllm", "max_model_len": 18_000},
    }
    write_sample_json_line(path, previous, {"sample": {"sample_id": "done"}})

    with pytest.raises(ValueError, match="run configuration mismatch"):
        completed_sample_ids(path, current)
