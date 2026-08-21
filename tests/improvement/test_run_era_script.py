from pathlib import Path

import pytest

from scripts.improvement.run_era import _load_sample_ids, _pending_records


def _record(sample_id: str) -> dict:
    return {"sample": {"sample_id": sample_id}}


def test_pending_records_filters_before_attention_work() -> None:
    records = [_record("keep"), _record("completed"), _record("outside")]

    pending = _pending_records(records, {"completed"}, {"keep", "completed"})

    assert set(pending) == {"keep"}


def test_load_sample_ids_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "subset_ids.txt"
    path.write_text("one\none\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates"):
        _load_sample_ids(path)


def test_load_sample_ids_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "subset_ids.txt"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        _load_sample_ids(path)
