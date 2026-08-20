from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.uq import run_split_uq


class FakeMethod:
    required_responses = "greedy_and_samples"
    runtime_config = {"name": "fake", "version": "test-v1"}

    def compute(self, *, question, greedy, samples):
        assert question == "Question"
        assert greedy.answer == "yes"
        assert [sample.answer for sample in samples] == ["one", "two"]
        assert samples[0].final_hidden == pytest.approx((1.0, 0.0))
        return {"valid": True, "error": None, "score": sum(sample.mean_log_prob for sample in samples)}


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    hidden_dir = tmp_path / "samples.hidden"
    hidden_dir.mkdir()
    torch.save(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16), hidden_dir / "one.pt")
    signal = {
        "token_count": 1,
        "sequence_log_prob": -0.1,
        "mean_log_prob": -0.1,
        "sampling_sequence_log_prob": -0.2,
    }
    greedy = tmp_path / "greedy.jsonl"
    greedy.write_text(
        json.dumps({"record_type": "run", "run": {"dataset": "test", "model_family": "fake"}})
        + "\n"
        + json.dumps({"record_type": "sample", "sample": {"sample_id": "one", "question": "Question"}, "greedy": {"answer": "yes", "signals": signal, "sections_valid": True}})
        + "\n"
    )
    samples = tmp_path / "samples.jsonl"
    samples.write_text(
        json.dumps({"record_type": "run", "run": {"dataset": "test", "model_family": "fake"}})
        + "\n"
        + json.dumps(
            {
                "record_type": "sample",
                "sample": {"sample_id": "one", "question": "Question"},
                "samples": [
                    {"answer": "one", "signals": signal, "hidden_state_index": 0, "sections_valid": True},
                    {"answer": "two", "signals": signal, "hidden_state_index": 1, "sections_valid": True},
                ],
                "hidden_states": {"path": "samples.hidden/one.pt", "shape": [2, 2], "dtype": "float16"},
            }
        )
        + "\n"
    )
    return greedy, samples


def test_split_uq_reads_separate_greedy_samples_and_hidden(tmp_path: Path) -> None:
    greedy, samples = inputs(tmp_path)
    output = tmp_path / "uq.jsonl"

    assert run_split_uq(greedy_input=greedy, sample_input=samples, output=output, methods=(FakeMethod(),)) == (1, 0)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[1]["uq"]["fake"]["score"] == pytest.approx(-0.2)
    assert run_split_uq(greedy_input=greedy, sample_input=samples, output=output, methods=(FakeMethod(),)) == (0, 1)


def test_split_uq_rejects_mismatched_inputs(tmp_path: Path) -> None:
    greedy, samples = inputs(tmp_path)
    rows = [json.loads(line) for line in samples.read_text().splitlines()]
    rows[0]["run"]["dataset"] = "other"
    samples.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="different datasets"):
        run_split_uq(greedy_input=greedy, sample_input=samples, output=tmp_path / "uq.jsonl", methods=(FakeMethod(),))
