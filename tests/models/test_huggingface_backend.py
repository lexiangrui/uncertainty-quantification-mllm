import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.generation.prompt import GenerationPrompt
from src.models.base import GeneratedResponse, GenerationRequest
from src.models.huggingface import HuggingFaceReplayBackend


def _request(name: str, *, image=None, role="sample") -> GenerationRequest:
    return GenerationRequest(
        request_id=name,
        sample_id=name,
        role=role,
        draw_index=0 if role == "sample" else None,
        seed=1,
        image=image,
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
    exception_contexts: list[BaseException | None] = []

    def replay(requests, token_sequences):
        calls.append(len(requests))
        exception_contexts.append(sys.exception())
        if len(requests) > 2:
            raise torch.OutOfMemoryError()
        return {request.request_id: _response(request) for request in requests}

    backend._teacher_force_batch = replay
    requests = [_request(str(index)) for index in range(5)]
    values = backend.teacher_force_responses(requests, [(1,)] * 5)
    assert set(values) == {"0", "1", "2", "3", "4"}
    assert calls == [5, 2, 3, 1, 2]
    assert exception_contexts == [None] * len(calls)


def test_replay_rejects_mismatched_inputs() -> None:
    backend = object.__new__(HuggingFaceReplayBackend)
    with pytest.raises(ValueError, match="equal length"):
        backend.teacher_force_responses([_request("one")], [])


class FakeTokenizer:
    def __init__(self):
        self.rendered = None
        self.padding_side = "right"

    def __call__(self, rendered, **_kwargs):
        self.rendered = rendered
        return {
            "input_ids": torch.tensor([[1, 2], [3, 4]][: len(rendered)]),
            "attention_mask": torch.ones((len(rendered), 2), dtype=torch.long),
        }

    def convert_tokens_to_ids(self, token):
        assert token == "<IMG_CONTEXT>"
        return 99


def test_original_internvl_builds_text_only_inputs_without_pixels() -> None:
    backend = object.__new__(HuggingFaceReplayBackend)
    backend.model = object()
    backend.processor = FakeTokenizer()
    backend.device = torch.device("cpu")

    inputs = backend._original_batch_inputs([_request("text")])

    assert set(inputs) == {"input_ids", "attention_mask"}
    assert "<img>" not in backend.processor.rendered[0]
    assert "Question" in backend.processor.rendered[0]


def test_original_internvl_rejects_mixed_modality_batch() -> None:
    backend = object.__new__(HuggingFaceReplayBackend)
    backend.model = object()
    backend.processor = FakeTokenizer()
    backend.device = torch.device("cpu")

    with pytest.raises(ValueError, match="cannot mix"):
        backend._original_batch_inputs(
            [_request("image", image=Image.new("RGB", (2, 2))), _request("text")]
        )


def test_original_internvl_uses_dynamic_tiles_and_matching_context_tokens() -> None:
    config = SimpleNamespace(
        force_image_size=448,
        dynamic_image_size=True,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        use_thumbnail=True,
    )
    backend = object.__new__(HuggingFaceReplayBackend)
    backend.model = SimpleNamespace(config=config, num_image_token=2)
    backend.processor = FakeTokenizer()
    backend.device = torch.device("cpu")

    inputs = backend._original_batch_inputs(
        [_request("image", image=Image.new("RGB", (896, 448)))]
    )

    assert inputs["pixel_values"].shape == (3, 3, 448, 448)
    assert inputs["image_flags"].shape == (3, 1)
    assert backend.processor.rendered[0].count("<IMG_CONTEXT>") == 6


def test_original_internvl_text_replay_calls_language_model_directly() -> None:
    marker = object()

    class LanguageModel:
        def __call__(self, **kwargs):
            assert "pixel_values" not in kwargs
            assert kwargs["output_hidden_states"] is False
            return marker

    backend = object.__new__(HuggingFaceReplayBackend)
    backend.family = "internvl3_5_original"
    backend.model = SimpleNamespace(
        get_base_model=lambda: SimpleNamespace(language_model=LanguageModel())
    )

    assert backend._replay_forward({"input_ids": torch.tensor([[1]])}) is marker


def test_semantic_embedding_module_returns_decoder_body() -> None:
    decoder = object()
    backend = object.__new__(HuggingFaceReplayBackend)
    backend.model = SimpleNamespace(
        get_base_model=lambda: SimpleNamespace(
            language_model=SimpleNamespace(model=decoder)
        )
    )

    assert backend._semantic_embedding_module() is decoder


def test_replay_hooks_only_final_decoder_hidden_for_samples() -> None:
    class Decoder(torch.nn.Module):
        def forward(self, input_ids):
            hidden = input_ids.unsqueeze(-1).float().repeat(1, 1, 3)
            return SimpleNamespace(last_hidden_state=hidden)

    class LanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Decoder()

    class ReplayModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel()

        def forward(self, input_ids, **kwargs):
            assert kwargs["output_hidden_states"] is False
            hidden = self.language_model.model(input_ids).last_hidden_state
            logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
            return SimpleNamespace(logits=logits, last_hidden_state=hidden)

    backend = object.__new__(HuggingFaceReplayBackend)
    backend.family = "llava_1_5"
    backend.model = ReplayModel()
    backend.processor = SimpleNamespace(
        pad_token_id=0,
        eos_token_id=7,
        decode=lambda *_args, **_kwargs: "answer",
    )
    backend.device = torch.device("cpu")
    backend._batch_inputs = lambda _requests: {
        "input_ids": torch.tensor([[5, 6]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
    }

    sample = backend._teacher_force_batch(
        [_request("sample")], [(1, 2)]
    )["sample"]
    greedy = backend._teacher_force_batch(
        [_request("greedy", role="greedy")], [(1, 2)]
    )["greedy"]

    assert sample.hidden_steps is not None
    assert sample.hidden_steps.shape == (2, 3)
    assert torch.equal(sample.hidden_steps[:, 0], torch.tensor([1.0, 2.0]))
    assert greedy.hidden_steps is None
