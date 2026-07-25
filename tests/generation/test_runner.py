from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from src.datasets.base import BenchmarkSample
from src.generation.runner import run_generation
from src.models.base import GeneratedResponse, GenerationBackend, GenerationRequest


VALID = (
    "<vision>The object is visible.</vision>"
    "<reasoning>The evidence supports the answer.</reasoning>"
    "<answer>yes</answer>"
)
INVALID = "yes"


class FakeDynamicBackend(GenerationBackend):
    model_id = "fake-model"

    def __init__(self, invalid_first_sample: bool = False, invalid_greedy: bool = False):
        self.invalid_first_sample = invalid_first_sample
        self.invalid_greedy = invalid_greedy
        self.calls: list[list[GenerationRequest]] = []

    @property
    def runtime_config(self):
        return {"engine": "fake-dynamic", "max_num_seqs": 5}

    def decode_generated_tokens(self, token_ids):
        return "".join(chr(value) for value in token_ids).strip()

    @staticmethod
    def response(text: str) -> GeneratedResponse:
        token_ids = tuple(ord(value) for value in text)
        return GeneratedResponse(
            text=text,
            token_ids=token_ids,
            token_log_probs=tuple(-0.1 for _ in token_ids),
            sampling_token_log_probs=tuple(-0.1 for _ in token_ids),
            finish_reason="stop",
        )

    def generate_requests(self, requests, *, max_new_tokens):
        assert max_new_tokens == 256
        assert len(requests) <= 5
        assert len({request.role for request in requests}) == 1
        self.calls.append(list(requests))
        values = {}
        for request in requests:
            invalid = self.invalid_greedy and request.role == "greedy"
            invalid |= (
                self.invalid_first_sample
                and request.role == "sample"
                and request.draw_index == 0
                and request.request_id.endswith("attempt-1")
            )
            values[request.request_id] = self.response(INVALID if invalid else VALID)
        return values


def sample(sample_id: str = "dataset-1") -> BenchmarkSample:
    return BenchmarkSample(
        sample_id=sample_id,
        group_id=sample_id,
        dataset="mmvet",
        split="test",
        question="Is it visible?",
        references=("yes",),
        image=Image.new("RGB", (2, 2)),
    )


def read(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return rows[0]["run"], rows[1:]


def kwargs(tmp_path: Path, backend: GenerationBackend) -> dict:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    source = tmp_path / "source.parquet"
    source.touch()
    return {
        "backend": backend,
        "family": "llava_1_5",
        "model_path": model,
        "dataset": "mmvet",
        "dataset_source": source,
        "output": tmp_path / "responses.jsonl",
        "max_new_tokens": 256,
        "num_samples": 5,
        "seed": 42,
        "limit": None,
        "reject_resample_k": 10,
        "max_batch_size": 5,
        "request_window_samples": 16,
    }


def test_dynamic_generation_writes_answers_tokens_and_resumes(monkeypatch, tmp_path):
    backend = FakeDynamicBackend()
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample()]))
    options = kwargs(tmp_path, backend)

    assert run_generation(**options) == (1, 0)
    run, records = read(options["output"])
    record = records[0]
    assert run["scheduler"]["max_batch_size"] == 5
    assert run["scheduler"]["type"] == "transformers_role_separated_dynamic_batching"
    assert run["scheduler"]["mixed_greedy_and_sampling"] is False
    assert run["greedy"]["retry"] is False
    assert run["hidden_state_execution"] == "separate"
    assert "hidden_states" not in record
    assert record["greedy"]["sections_valid"] is True
    assert "reject_resample" not in record["greedy"]
    assert len(record["samples"]) == 5
    assert all(item["reject_resample"]["attempts_used"] == 1 for item in record["samples"])
    token_path = options["output"].parent / record["generation_tokens"]["path"]
    tokens = torch.load(token_path, weights_only=True)
    assert set(tokens) == {"greedy", "sample_0", "sample_1", "sample_2", "sample_3", "sample_4"}
    assert run_generation(**options) == (0, 1)


def test_only_sampled_responses_are_retried(monkeypatch, tmp_path):
    backend = FakeDynamicBackend(invalid_first_sample=True, invalid_greedy=True)
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample()]))
    options = kwargs(tmp_path, backend)

    run_generation(**options)
    _, records = read(options["output"])
    record = records[0]
    assert record["greedy"]["sections_valid"] is False
    greedy_requests = [
        request for call in backend.calls for request in call if request.role == "greedy"
    ]
    assert len(greedy_requests) == 1
    retried = record["samples"][0]
    assert retried["sections_valid"] is True
    assert retried["reject_resample"] == {
        "max_attempts": 10,
        "attempts_used": 2,
        "rejected_count": 1,
        "accepted": True,
    }
    assert len(retried["attempt_seeds"]) == 2
    assert record["reject_resample_summary"]["total_attempts"] == 6
    assert record["reject_resample_summary"]["rejected_attempts"] == 1


def test_request_window_separates_decoding_roles_into_dynamic_batches(monkeypatch, tmp_path):
    backend = FakeDynamicBackend()
    values = [sample("one"), sample("two")]
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter(values))
    options = kwargs(tmp_path, backend)
    options["request_window_samples"] = 2

    assert run_generation(**options) == (2, 0)
    first = backend.calls[0]
    assert {request.sample_id for request in first} == {"one", "two"}
    assert {request.role for request in first} == {"greedy"}
    assert all(len(call) <= 5 for call in backend.calls)
    assert all(len({request.role for request in call}) == 1 for call in backend.calls)
    assert sum(len(call) for call in backend.calls if call[0].role == "sample") == 10
