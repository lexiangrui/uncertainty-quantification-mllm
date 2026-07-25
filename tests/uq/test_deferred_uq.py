from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.uq import run_deferred_uq


class FakeMethod:
    required_responses = "greedy_and_samples"
    runtime_config = {"name": "fake", "version": "test-v1"}

    def compute(self, *, question, greedy, samples):
        assert question == "Question"
        assert greedy.answer == "yes"
        assert [sample.answer for sample in samples] == ["one", "two"]
        assert samples[0].final_hidden == pytest.approx((1.0, 0.0))
        return {"valid": True, "error": None, "score": sum(sample.mean_log_prob for sample in samples)}


def generation_file(tmp_path: Path) -> Path:
    hidden_dir = tmp_path / "generation.hidden"
    hidden_dir.mkdir()
    torch.save(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16), hidden_dir / "one.pt")
    signal = {
        "token_count": 1,
        "sequence_log_prob": -0.1,
        "mean_log_prob": -0.1,
        "sampling_sequence_log_prob": -0.2,
    }
    rows = [
        {"record_type": "run", "run": {"generation_output_version": "responses-jsonl-hidden-pt-v1"}},
        {
            "record_type": "sample",
            "sample": {"sample_id": "one", "question": "Question"},
            "greedy": {"answer": "yes", "signals": signal},
            "samples": [
                {"answer": "one", "signals": signal, "hidden_state_index": 0},
                {"answer": "two", "signals": signal, "hidden_state_index": 1},
            ],
            "hidden_states": {"path": "generation.hidden/one.pt", "shape": [2, 2], "dtype": "float16"},
        },
    ]
    path = tmp_path / "generation.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_deferred_uq_reads_hidden_sidecar_and_resumes(tmp_path: Path) -> None:
    generation = generation_file(tmp_path)
    output = tmp_path / "uq.jsonl"

    assert run_deferred_uq(
        generation_input=generation, output=output, methods=(FakeMethod(),)
    ) == (1, 0)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["run"]["uq_output_version"] == "deferred-uq-v1"
    assert rows[1]["sample"] == {"sample_id": "one"}
    assert rows[1]["uq"]["fake"]["score"] == pytest.approx(-0.2)
    assert run_deferred_uq(
        generation_input=generation, output=output, methods=(FakeMethod(),)
    ) == (0, 1)


def test_deferred_uq_rejects_hidden_path_escape(tmp_path: Path) -> None:
    generation = generation_file(tmp_path)
    rows = [json.loads(line) for line in generation.read_text().splitlines()]
    rows[1]["hidden_states"]["path"] = "../outside.pt"
    generation.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="escapes"):
        run_deferred_uq(
            generation_input=generation,
            output=tmp_path / "uq.jsonl",
            methods=(FakeMethod(),),
        )
