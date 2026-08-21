from pathlib import Path

import pytest
import torch

from src.generation.prompt import GenerationPrompt
from src.models.base import GeneratedResponse, GenerationRequest
from src.models.huggingface import HuggingFaceReplayBackend


def _request(name: str) -> GenerationRequest:
    return GenerationRequest(
        request_id=name,
        sample_id=name,
        role="sample",
        draw_index=0,
        seed=1,
        image=None,
        prompt=GenerationPrompt(system="", user="Question"),
    )


def _response(request: GenerationRequest) -> GeneratedResponse:
    return GeneratedResponse(
        text="answer",
        token_ids=(1,),
        token_log_probs=(-0.1,),
        sampling_token_log_probs=(-0.1,),
        rng_seed=request.seed,
    )


def test_qwen_split_device_map_is_recorded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QWEN_DEVICE_MAP", "vision_language_split")
    backend = HuggingFaceReplayBackend(
        "qwen2_5_vl",
        tmp_path,
        attn_implementation="flash_attention_2",
        adapter_path=None,
    )
    assert backend.runtime_config["engine"] == "hf_replay"
    assert backend.runtime_config["device_map"] == "vision_language_split"


def test_replay_oom_recursively_splits() -> None:
    backend = object.__new__(HuggingFaceReplayBackend)
    calls: list[int] = []

    def replay(requests, token_sequences):
        calls.append(len(requests))
        if len(requests) > 2:
            raise torch.OutOfMemoryError()
        return {request.request_id: _response(request) for request in requests}

    backend._teacher_force_batch = replay
    requests = [_request(str(index)) for index in range(5)]
    values = backend.teacher_force_responses(requests, [(1,)] * 5)
    assert set(values) == {"0", "1", "2", "3", "4"}
    assert calls == [5, 2, 3, 1, 2]


def test_replay_rejects_mismatched_inputs() -> None:
    backend = object.__new__(HuggingFaceReplayBackend)
    with pytest.raises(ValueError, match="equal length"):
        backend.teacher_force_responses([_request("one")], [])
