from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch
from PIL import Image

from src.datasets.base import BenchmarkSample
from src.generation.runner import _hidden_sidecar_path, run_generation
from src.generation.prompt import XML_LORA_PROMPT_SHA256
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
    def response(text: str, *, with_hidden: bool) -> GeneratedResponse:
        token_ids = tuple(ord(value) for value in text)
        return GeneratedResponse(
            text=text,
            token_ids=token_ids,
            token_log_probs=tuple(-0.1 for _ in token_ids),
            sampling_token_log_probs=tuple(-0.1 for _ in token_ids),
            finish_reason="stop",
            hidden_steps=(
                torch.tensor(
                    [[float(index) for index in range(4)] for _ in token_ids],
                    dtype=torch.float32,
                )
                if with_hidden
                else None
            ),
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
            values[request.request_id] = self.response(
                INVALID if invalid else VALID,
                with_hidden=request.role == "sample",
            )
        return values


class FakeVllmBackend(FakeDynamicBackend):
    @property
    def runtime_config(self):
        return {"engine": "vllm", "max_num_seqs": 4}

    def generate_requests(self, requests, *, max_new_tokens):
        assert max_new_tokens == 256
        assert len(requests) <= 16
        assert len({request.role for request in requests}) == 1
        self.calls.append(list(requests))
        return {
            request.request_id: self.response(VALID, with_hidden=False)
            for request in requests
        }


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
        "phase": "samples",
    }


def test_sample_generation_writes_answers_tokens_and_resumes(monkeypatch, tmp_path):
    backend = FakeDynamicBackend()
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample()]))
    options = kwargs(tmp_path, backend)

    assert run_generation(**options) == (1, 0)
    run, records = read(options["output"])
    record = records[0]
    assert run["scheduler"]["max_batch_size"] == 5
    assert run["scheduler"]["type"] == "transformers_role_separated_dynamic_batching"
    assert run["scheduler"]["mixed_greedy_and_sampling"] is False
    assert run["generation_phase"] == "samples"
    assert run["hidden_state_execution"] == "inline_sample_answer_last_token"
    assert run["prompt_sha256"] == XML_LORA_PROMPT_SHA256
    assert record["hidden_states"]["shape"] == [5, 4]
    assert "greedy" not in record
    assert len(record["samples"]) == 5
    assert all(item["reject_resample"]["attempts_used"] == 1 for item in record["samples"])
    token_path = options["output"].parent / record["generation_tokens"]["path"]
    tokens = torch.load(token_path, weights_only=True)
    assert set(tokens) == {"sample_0", "sample_1", "sample_2", "sample_3", "sample_4"}
    hidden_path = options["output"].parent / record["hidden_states"]["path"]
    hidden = torch.load(hidden_path, weights_only=True)
    assert hidden.shape == (5, 4)
    assert all(item["hidden_state_index"] == index for index, item in enumerate(record["samples"]))
    assert run_generation(**options) == (0, 1)


def test_greedy_generation_writes_only_greedy_response(monkeypatch, tmp_path):
    backend = FakeDynamicBackend()
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample()]))
    options = kwargs(tmp_path, backend)
    options.update(phase="greedy", num_samples=0)

    assert run_generation(**options) == (1, 0)
    run, records = read(options["output"])
    record = records[0]
    assert run["generation_phase"] == "greedy"
    assert run["hidden_state_execution"] == "not_collected"
    assert record["samples"] == []
    assert record["greedy"]["answer"] == "yes"
    assert "hidden_states" not in record
    token_path = options["output"].parent / record["generation_tokens"]["path"]
    assert set(torch.load(token_path, weights_only=True)) == {"greedy"}


def test_only_sampled_responses_are_retried(monkeypatch, tmp_path):
    backend = FakeDynamicBackend(invalid_first_sample=True, invalid_greedy=True)
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample()]))
    options = kwargs(tmp_path, backend)

    run_generation(**options)
    _, records = read(options["output"])
    record = records[0]
    greedy_requests = [
        request for call in backend.calls for request in call if request.role == "greedy"
    ]
    assert not greedy_requests
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


def test_request_window_batches_samples_without_greedy(monkeypatch, tmp_path):
    backend = FakeDynamicBackend()
    values = [sample("one"), sample("two")]
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter(values))
    options = kwargs(tmp_path, backend)
    options["request_window_samples"] = 2

    assert run_generation(**options) == (2, 0)
    assert all(len(call) <= 5 for call in backend.calls)
    assert all(len({request.role for request in call}) == 1 for call in backend.calls)


def test_vllm_uses_continuous_dispatch_and_pending_backfill_metadata(monkeypatch, tmp_path):
    backend = FakeVllmBackend()
    values = [sample(str(index)) for index in range(12)]
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter(values))
    options = kwargs(tmp_path, backend)
    options.update(num_samples=1, request_window_samples=12)

    assert run_generation(**options) == (12, 0)
    run, _ = read(options["output"])
    assert run["scheduler"]["type"] == "vllm_continuous_batching"
    assert run["scheduler"]["max_batch_size"] == 16
    assert run["hidden_state_execution"] == "pending_hf_teacher_forcing_backfill"
    assert len(backend.calls[0]) == 12


def test_sample_id_filter_runs_only_requested_records(monkeypatch, tmp_path):
    backend = FakeDynamicBackend()
    values = [sample("one"), sample("two")]
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter(values))
    options = kwargs(tmp_path, backend)
    options["sample_ids"] = {"two"}

    assert run_generation(**options) == (1, 0)
    run, records = read(options["output"])
    assert run["sample_filter"] == ["two"]
    assert [record["sample"]["sample_id"] for record in records] == ["two"]
    assert all(call[0].role == "sample" for call in backend.calls)
    assert sum(len(call) for call in backend.calls) == options["num_samples"]
    assert {request.sample_id for request in backend.calls[0]} == {"two"}


def test_generation_prompt_sha_matches_file() -> None:
    path = Path(__file__).resolve().parents[2] / "prompts" / "generation" / "xml_lora_zero_shot.md"
    text = path.read_text(encoding="utf-8").strip()
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == XML_LORA_PROMPT_SHA256


def test_production_hidden_paths_follow_model_and_dataset_layout() -> None:
    output = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "generation"
        / "llava"
        / "samples"
        / "mmvet.jsonl"
    )
    path, descriptor = _hidden_sidecar_path(output, "mmvet-1")
    assert path == output.parents[4] / "results" / "hidden" / "llava" / "mmvet" / "4520f8b99db1622b.pt"
    assert descriptor == {
        "path": "llava/mmvet/4520f8b99db1622b.pt",
        "storage": "results_hidden",
    }
