from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.datasets.base import BenchmarkSample
from src.generation.runner import run_generation
from src.models.base import GeneratedResponse, GenerationBackend


RESPONSE = (
    "<vision>The relevant visual evidence is present.</vision>"
    "<reasoning>The evidence supports the answer.</reasoning>"
    "<answer>yes</answer>"
)


def read_output(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["record_type"] == "run"
    return rows[0]["run"], rows[1:]


class FakeBackend(GenerationBackend):
    model_id = "fake-model"

    def decode_generated_tokens(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)

    @staticmethod
    def response(text):
        token_ids = tuple(ord(character) for character in text)
        return GeneratedResponse(
            text=text,
            token_ids=token_ids,
            token_log_probs=tuple(-0.1 for _ in text),
            sampling_token_log_probs=tuple(-0.2 for _ in text),
            final_hidden=(1.0, 0.0),
        )

    def generate(
        self,
        image,
        prompt,
        *,
        do_sample,
        temperature,
        max_new_tokens,
        num_return_sequences,
    ):
        return [self.response(RESPONSE)] * num_return_sequences


class FakeUQ:
    required_responses = "samples"
    runtime_config = {"name": "fake_uq", "version": "test-v1"}

    def compute(self, *, question, greedy, samples):
        assert question
        assert greedy.answer == "yes"
        assert [item.answer for item in samples] == ["yes"] * 10
        assert [item.mean_log_prob for item in samples] == pytest.approx([-0.1] * 10)
        return {"valid": True, "error": None, "score": 0.25}


class DatasetCalibratedUQ:
    required_responses = "greedy"
    runtime_config = {"name": "calibrated_uq", "version": "test-v1"}

    def compute(self, *, question, greedy, samples):
        return {
            "valid": True,
            "error": None,
            "component": float(question),
            "calibration": None,
            "score": None,
        }

    @staticmethod
    def finalize(values):
        valid = [value for value in values if value.get("valid") is True]
        if not valid:
            return
        calibration = sum(value["component"] for value in valid) / len(valid)
        for value in valid:
            value["calibration"] = calibration
            value["score"] = value["component"] + calibration


def test_generation_record_and_resume(monkeypatch, tmp_path: Path) -> None:
    sample = BenchmarkSample(
        sample_id="dataset-1",
        group_id="dataset-1",
        dataset="mmvet",
        split="test",
        question="Is it visible?",
        references=("yes",),
        image=Image.new("RGB", (2, 2)),
    )
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample]))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.arrow"
    source.touch()
    output = tmp_path / "responses.jsonl"
    kwargs = dict(
        backend=FakeBackend(),
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=10,
        seed=42,
        limit=None,
        uq_methods=(FakeUQ(),),
    )

    assert run_generation(**kwargs) == (1, 0)
    run, records = read_output(output)
    record = records[0]
    assert record["record_type"] == "sample"
    assert "run" not in record
    assert record["sample"]["has_image"] is True
    assert record["greedy"]["answer"] == "yes"
    assert record["greedy"]["sections_valid"] is True
    assert "generation_trace" not in record["greedy"]
    assert "answer_trace" not in record["greedy"]
    assert len(record["samples"]) == 10
    assert all("generation_trace" not in item for item in record["samples"])
    assert all("answer_trace" not in item for item in record["samples"])
    assert record["uq"]["fake_uq"]["score"] == 0.25
    assert run["uq_execution"] == "online-compatibility"
    hidden = record["hidden_states"]
    hidden_path = output.parent / hidden["path"]
    tensor = torch.load(hidden_path, weights_only=True)
    assert tensor.dtype == torch.float16
    assert list(tensor.shape) == [10, 2]
    assert [item["hidden_state_index"] for item in record["samples"]] == list(range(10))
    assert record["greedy"]["signals"]["mean_log_prob"] == pytest.approx(-0.1)
    assert all(item["signals"]["mean_log_prob"] == pytest.approx(-0.1) for item in record["samples"])
    assert run_generation(**kwargs) == (0, 1)
    assert len(output.read_text().splitlines()) == 2


def test_resume_rejects_changed_configuration(monkeypatch, tmp_path: Path) -> None:
    sample = BenchmarkSample(
        sample_id="dataset-1",
        group_id="dataset-1",
        dataset="mmvet",
        split="test",
        question="Question",
        references=("answer",),
        image=None,
    )
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample]))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.arrow"
    source.touch()
    output = tmp_path / "responses.jsonl"
    base = dict(
        backend=FakeBackend(),
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=10,
        seed=42,
        limit=None,
        uq_methods=(FakeUQ(),),
    )
    run_generation(**base)
    base["seed"] = 43
    try:
        run_generation(**base)
    except ValueError as error:
        assert "configuration mismatch" in str(error)
    else:
        raise AssertionError("changed run configuration must fail")


def test_sampling_rejects_invalid_response_then_accepts_retry(
    monkeypatch, tmp_path: Path
) -> None:
    invalid = "a direct answer without blocks"

    class RetryBackend(FakeBackend):
        def __init__(self):
            self.sample_calls = 0
            self.sample_seeds = []

        def generate(self, *args, do_sample, num_return_sequences, **kwargs):
            if not do_sample:
                return [self.response(RESPONSE)]
            self.sample_calls += 1
            self.sample_seeds.append(torch.initial_seed())
            text = invalid if self.sample_calls == 1 else RESPONSE
            return [self.response(text)] * num_return_sequences

    sample = BenchmarkSample(
        sample_id="dataset-1",
        group_id="dataset-1",
        dataset="mmvet",
        split="test",
        question="Question",
        references=("yes",),
        image=None,
    )
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample]))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.parquet"
    source.touch()
    output = tmp_path / "responses.jsonl"
    backend = RetryBackend()

    run_generation(
        backend=backend,
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=1,
        seed=42,
        limit=None,
        uq_methods=(),
        reject_resample_k=3,
    )

    _, records = read_output(output)
    record = records[0]
    retained = record["samples"][0]
    assert backend.sample_calls == 2
    assert len(set(backend.sample_seeds)) == 2
    assert retained["sections_valid"] is True
    assert retained["reject_resample"] == {
        "max_attempts": 3,
        "attempts_used": 2,
        "rejected_count": 1,
        "accepted": True,
    }
    assert retained["seed"] == backend.sample_seeds[-1]
    assert record["reject_resample_summary"] == {
        "max_attempts": 3,
        "retained_samples": 1,
        "accepted_samples": 1,
        "failed_samples": 0,
        "total_attempts": 2,
        "rejected_attempts": 1,
    }


def test_greedy_rejects_invalid_response_then_recovers_with_sampling(
    monkeypatch, tmp_path: Path
) -> None:
    class GreedyRetryBackend(FakeBackend):
        def __init__(self):
            self.calls = []

        def generate(self, *args, do_sample, temperature, num_return_sequences, **kwargs):
            self.calls.append((do_sample, temperature, num_return_sequences))
            if len(self.calls) == 1:
                return [self.response("invalid greedy response")]
            return [self.response(RESPONSE)] * num_return_sequences

    sample = BenchmarkSample(
        sample_id="dataset-1",
        group_id="dataset-1",
        dataset="mmvet",
        split="test",
        question="Question",
        references=("yes",),
        image=None,
    )
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample]))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.parquet"
    source.touch()
    output = tmp_path / "responses.jsonl"
    backend = GreedyRetryBackend()

    run_generation(
        backend=backend,
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=1,
        seed=42,
        limit=None,
        uq_methods=(),
        greedy_reject_resample_k=3,
        greedy_recovery_temperature=0.2,
    )

    run, records = read_output(output)
    greedy = records[0]["greedy"]
    assert backend.calls[:2] == [(False, None, 1), (True, 0.2, 1)]
    assert greedy["sections_valid"] is True
    assert greedy["reject_resample"] == {
        "max_attempts": 3,
        "attempts_used": 2,
        "rejected_count": 1,
        "accepted": True,
        "initial_strategy": "greedy",
        "accepted_strategy": "low_temperature_sampling",
    }
    assert run["greedy"]["reject_resample_k"] == 3
    assert run["greedy"]["recovery_temperature"] == 0.2


def test_sampling_batches_initial_generation(monkeypatch, tmp_path: Path) -> None:
    class BatchBackend(FakeBackend):
        def __init__(self):
            self.sample_batch_sizes = []

        def generate(self, *args, do_sample, num_return_sequences, **kwargs):
            if do_sample:
                self.sample_batch_sizes.append(num_return_sequences)
            return [self.response(RESPONSE)] * num_return_sequences

    sample = BenchmarkSample(
        sample_id="dataset-1",
        group_id="dataset-1",
        dataset="mmvet",
        split="test",
        question="Question",
        references=("yes",),
        image=None,
    )
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample]))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.parquet"
    source.touch()
    output = tmp_path / "responses.jsonl"
    backend = BatchBackend()

    run_generation(
        backend=backend,
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=5,
        seed=42,
        limit=None,
        uq_methods=(),
        sampling_batch_size=2,
    )

    run, records = read_output(output)
    assert backend.sample_batch_sizes == [2, 2, 1]
    assert len(records[0]["samples"]) == 5
    assert run["sampling"]["batch_size"] == 2


def test_sampling_keeps_last_response_after_retry_limit(
    monkeypatch, tmp_path: Path
) -> None:
    class AlwaysInvalidBackend(FakeBackend):
        def __init__(self):
            self.sample_calls = 0

        def generate(self, *args, do_sample, num_return_sequences, **kwargs):
            if not do_sample:
                return [self.response(RESPONSE)]
            self.sample_calls += 1
            return [self.response(f"invalid attempt {self.sample_calls}")] * num_return_sequences

    sample = BenchmarkSample(
        sample_id="dataset-1",
        group_id="dataset-1",
        dataset="mmvet",
        split="test",
        question="Question",
        references=("yes",),
        image=None,
    )
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample]))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.parquet"
    source.touch()
    output = tmp_path / "responses.jsonl"
    backend = AlwaysInvalidBackend()

    run_generation(
        backend=backend,
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=1,
        seed=42,
        limit=None,
        uq_methods=(),
        reject_resample_k=3,
    )

    _, records = read_output(output)
    record = records[0]
    retained = record["samples"][0]
    assert backend.sample_calls == 3
    assert retained["raw_response"] == "invalid attempt 3"
    assert retained["reject_resample"]["accepted"] is False
    assert retained["reject_resample"]["attempts_used"] == 3
    assert retained["reject_resample"]["rejected_count"] == 3
    assert len(record["samples"]) == 1
    assert record["reject_resample_summary"]["failed_samples"] == 1


def test_resume_rejects_changed_reject_resample_limit(
    monkeypatch, tmp_path: Path
) -> None:
    sample = BenchmarkSample(
        sample_id="dataset-1",
        group_id="dataset-1",
        dataset="mmvet",
        split="test",
        question="Question",
        references=("answer",),
        image=None,
    )
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample]))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.arrow"
    source.touch()
    output = tmp_path / "responses.jsonl"
    base = dict(
        backend=FakeBackend(),
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=10,
        seed=42,
        limit=None,
        uq_methods=(FakeUQ(),),
        reject_resample_k=2,
    )
    run_generation(**base)
    base["reject_resample_k"] = 3
    with pytest.raises(ValueError, match="configuration mismatch"):
        run_generation(**base)


def test_unseparated_response_is_preserved(monkeypatch, tmp_path: Path) -> None:
    class InvalidBackend(FakeBackend):
        def generate(self, *args, num_return_sequences, **kwargs):
            return [self.response("a direct answer without blocks")] * num_return_sequences

    sample = BenchmarkSample(
        sample_id="dataset-1",
        group_id="dataset-1",
        dataset="mmvet",
        split="test",
        question="Question",
        references=("answer",),
        image=None,
    )
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample]))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.parquet"
    source.touch()
    output = tmp_path / "responses.jsonl"
    run_generation(
        backend=InvalidBackend(),
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=10,
        seed=42,
        limit=None,
        uq_methods=(FakeUQ(),),
    )
    _, records = read_output(output)
    record = records[0]
    assert record["greedy"]["raw_response"] == "a direct answer without blocks"
    assert record["greedy"]["sections_valid"] is False
    assert record["greedy"]["answer"] is None
    assert all(item["sections_valid"] is False for item in record["samples"])
    assert record["uq"]["fake_uq"] == {
        "valid": False,
        "error": "one or more sampled responses cannot be separated into three parts",
        "score": None,
    }


def test_dataset_finalizer_recalibrates_old_records_after_resume(
    monkeypatch, tmp_path: Path
) -> None:
    samples = [
        BenchmarkSample(
            sample_id=f"dataset-{index}",
            group_id=f"dataset-{index}",
            dataset="mmvet",
            split="test",
            question=str(index),
            references=("yes",),
            image=None,
        )
        for index in (1, 3)
    ]
    visible = 1

    def dataset(*args):
        return iter(samples[:visible])

    monkeypatch.setattr("src.generation.runner.iter_dataset", dataset)
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.parquet"
    source.touch()
    output = tmp_path / "responses.jsonl"
    kwargs = dict(
        backend=FakeBackend(),
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=10,
        seed=42,
        limit=None,
        uq_methods=(DatasetCalibratedUQ(),),
    )

    assert run_generation(**kwargs) == (1, 0)
    _, records = read_output(output)
    first = records[0]
    assert first["uq"]["calibrated_uq"]["calibration"] == 1.0

    visible = 2
    assert run_generation(**kwargs) == (1, 1)
    run, records = read_output(output)
    assert [record["uq"]["calibrated_uq"]["calibration"] for record in records] == [
        2.0,
        2.0,
    ]
    assert [record["uq"]["calibrated_uq"]["score"] for record in records] == [
        3.0,
        5.0,
    ]
    assert run["dataset"] == "mmvet"
    assert all("run" not in record for record in records)


def test_dataset_finalizer_accepts_empty_dataset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter(()))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.parquet"
    source.touch()
    output = tmp_path / "responses.jsonl"

    result = run_generation(
        backend=FakeBackend(),
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=10,
        seed=42,
        limit=None,
        uq_methods=(DatasetCalibratedUQ(),),
    )

    assert result == (0, 0)
    assert not output.exists()


def test_generation_without_online_uq_writes_deferred_features(
    monkeypatch, tmp_path: Path
) -> None:
    sample = BenchmarkSample(
        sample_id="dataset-1",
        group_id="dataset-1",
        dataset="mmvet",
        split="test",
        question="Question",
        references=("yes",),
        image=None,
    )
    monkeypatch.setattr("src.generation.runner.iter_dataset", lambda *args: iter([sample]))
    model_path = tmp_path / "model"
    model_path.mkdir()
    source = tmp_path / "source.parquet"
    source.touch()
    output = tmp_path / "responses.jsonl"

    run_generation(
        backend=FakeBackend(),
        family="llava_1_5",
        model_path=model_path,
        dataset="mmvet",
        dataset_source=source,
        output=output,
        max_new_tokens=256,
        num_samples=2,
        seed=42,
        limit=None,
        uq_methods=(),
    )

    run, records = read_output(output)
    record = records[0]
    assert run["uq_execution"] == "deferred"
    assert run["uq_methods"] == []
    assert "uq" not in record
    tensor = torch.load(output.parent / record["hidden_states"]["path"], weights_only=True)
    assert tensor.shape == (2, 2)
