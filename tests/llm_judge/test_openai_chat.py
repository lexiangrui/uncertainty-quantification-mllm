from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.datasets.base import BenchmarkSample
from src.llm_judge.openai_chat import (
    JUDGE_SYSTEM_PROMPT,
    OpenAIChatJudge,
    build_openai_judge_messages,
    parse_openai_judge_response,
)
from src.llm_judge.runner import run_openai_judging


VALID_RESPONSE = json.dumps(
    {
        "analysis": "The answer is correct and the observations are grounded.",
        "correct": True,
        "rating": 4,
        "hallucination_types": [],
    }
)


def test_prompt_explicitly_requests_lowercase_json() -> None:
    assert "json" in JUDGE_SYSTEM_PROMPT


class FakeCompletions:
    def __init__(self, response: str = VALID_RESPONSE) -> None:
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self.response)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, response: str = VALID_RESPONSE) -> None:
        self.completions = FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def test_messages_include_data_url_only_when_image_exists() -> None:
    common = dict(
        dataset="vilp",
        question="How many?",
        references=["4"],
        vision="Four propellers are visible.",
        reasoning="Counting gives four.",
        answer="4",
    )
    with_image = build_openai_judge_messages(
        **common, image=Image.new("RGB", (2, 2), "red")
    )
    content = with_image[1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "text"
    assert "json" in content[1]["text"]

    without_image = build_openai_judge_messages(**common, image=None)
    assert [item["type"] for item in without_image[1]["content"]] == ["text"]


def test_client_uses_chat_completions_json_mode() -> None:
    client = FakeClient()
    judge = OpenAIChatJudge("judge-model", client=client)
    result = judge.judge(
        dataset="vilp",
        question="How many?",
        references=["4"],
        vision="Four propellers are visible.",
        reasoning="Counting gives four.",
        answer="4",
        image=None,
    )
    assert result.correct is True
    assert result.hallucination is False
    assert client.completions.kwargs["model"] == "judge-model"
    assert client.completions.kwargs["temperature"] == 0
    assert client.completions.kwargs["response_format"] == {"type": "json_object"}


def test_response_validation() -> None:
    result = parse_openai_judge_response(VALID_RESPONSE)
    assert result.rating == 4
    with pytest.raises(ValueError, match="inconsistent"):
        parse_openai_judge_response(
            '{"analysis":"bad","correct":false,"rating":2,"hallucination_types":[]}'
        )


def test_environment_is_required(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        OpenAIChatJudge("judge-model")


def test_runner_writes_and_resumes(monkeypatch, tmp_path: Path) -> None:
    sample = BenchmarkSample(
        sample_id="vilp-0-case1",
        group_id="vilp-0",
        dataset="vilp",
        split="test",
        question="How many?",
        references=("4",),
        image=Image.new("RGB", (2, 2)),
    )
    monkeypatch.setattr("src.llm_judge.runner.iter_dataset", lambda *args: iter([sample]))
    source = tmp_path / "source.parquet"
    source.touch()
    generation_input = tmp_path / "generation.jsonl"
    generation_run = {"prompt_version": "xml-lora-zero-shot-v1"}
    generation_input.write_text(
        json.dumps(
            {
                "run": generation_run,
                "sample": {"sample_id": sample.sample_id},
                "greedy": {
                    "sections_valid": True,
                    "raw_response": "response",
                    "vision": "Four propellers are visible.",
                    "reasoning": "Counting gives four.",
                    "answer": "4",
                },
            }
        )
        + "\n"
    )
    output = tmp_path / "judgements.jsonl"
    judge = OpenAIChatJudge("judge-model", client=FakeClient())
    kwargs = dict(
        judge=judge,
        dataset="vilp",
        dataset_source=source,
        generation_input=generation_input,
        output=output,
        limit=None,
    )
    assert run_openai_judging(**kwargs) == (1, 0)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["record_type"] == "run"
    record = rows[1]
    assert record["record_type"] == "sample"
    assert "run" not in record
    assert record["judge"]["valid"] is True
    assert record["judge"]["correct"] is True
    assert run_openai_judging(**kwargs) == (0, 1)


def test_runner_records_unseparated_input_without_api_call(monkeypatch, tmp_path: Path) -> None:
    sample = BenchmarkSample(
        sample_id="vilp-0-case1",
        group_id="vilp-0",
        dataset="vilp",
        split="test",
        question="How many?",
        references=("4",),
        image=None,
    )
    monkeypatch.setattr("src.llm_judge.runner.iter_dataset", lambda *args: iter([sample]))
    source = tmp_path / "source.parquet"
    source.touch()
    generation_input = tmp_path / "generation.jsonl"
    generation_input.write_text(
        json.dumps(
            {
                "record_type": "run",
                "run": {"prompt_version": "xml-lora-zero-shot-v1"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "record_type": "sample",
                "sample": {"sample_id": sample.sample_id},
                "greedy": {
                    "sections_valid": False,
                    "raw_response": "4",
                    "vision": None,
                    "reasoning": None,
                    "answer": None,
                },
            }
        )
        + "\n"
    )
    output = tmp_path / "judgements.jsonl"
    client = FakeClient()
    judge = OpenAIChatJudge("judge-model", client=client)
    assert run_openai_judging(
        judge=judge,
        dataset="vilp",
        dataset_source=source,
        generation_input=generation_input,
        output=output,
        limit=None,
    ) == (1, 0)
    assert client.completions.kwargs is None
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[1]["judge"]["valid"] is False
