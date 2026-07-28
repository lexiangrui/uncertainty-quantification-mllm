from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_teacher_data.py"
SPEC = importlib.util.spec_from_file_location("generate_teacher_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_generate_one_accepts_valid_response(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(MODULE, "build_messages", lambda *args: [])
    monkeypatch.setattr(
        MODULE,
        "request_teacher_payload",
        lambda *args: {"vision": "A blue cup is visible.", "reasoning": "The cup is blue.", "answer": "blue"},
    )
    row = {
        "question_id": 1,
        "question": "What color is the cup?",
        "answer": "blue",
        "agreement": 10,
        "image_file": "one.jpg",
    }
    status, output = MODULE.generate_one(object(), "qwen3.7-plus", row, tmp_path, "prompt", [])
    assert status == "accepted"
    assert output["teacher"]["answer"] == "blue"


def test_generate_one_records_api_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(MODULE, "build_messages", lambda *args: [])

    def fail(*args):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(MODULE, "request_teacher_payload", fail)
    row = {
        "question_id": 2,
        "question": "What color is the cup?",
        "answer": "blue",
        "agreement": 10,
        "image_file": "two.jpg",
    }
    status, output = MODULE.generate_one(object(), "qwen3.7-plus", row, tmp_path, "prompt", [])
    assert status == "rejected"
    assert output["reason"] == "RuntimeError: rate limited"
