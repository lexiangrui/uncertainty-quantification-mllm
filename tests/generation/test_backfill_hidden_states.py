from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.generation.backfill_hidden_states import (
    BACKFILL_SOURCE,
    _answer_hidden_index,
    _answer_last_generated_token_offset,
    _build_prompt_text,
    _current_sidecar,
    backfill_hidden_states,
)
from src.generation.parser import answer_character_span


VALID = (
    "<vision>The object is visible.</vision>"
    "<reasoning>The evidence supports it.</reasoning>"
    "<answer>yes</answer>"
)


class CharacterTokenizer:
    pad_token_id = 0
    padding_side = "right"

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(value) for value in token_ids)


class PromptProcessor:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": False}
        assert all(part["type"] != "image" for part in messages[0]["content"])
        return messages[0]["content"][0]["text"] + "\nASSISTANT: "


class EmptyBackend:
    family = "llava_1_5"
    adapter_path = None
    device = model = processor = None

    def _load(self):
        return None


def test_answer_last_token_uses_generated_token_sidecar() -> None:
    token_ids = [ord(value) for value in VALID]
    _, answer_end = answer_character_span(VALID)
    assert (
        _answer_last_generated_token_offset(CharacterTokenizer(), token_ids, VALID)
        == answer_end - 1
    )


def test_hidden_index_is_stable_under_image_expansion_and_right_padding() -> None:
    # 600 prompt positions after image expansion, 10 generated positions,
    # then 3 right-padding positions. Generated offset 6 must map to 606.
    assert _answer_hidden_index(613, 3, 10, 6) == 606


def test_text_only_prompt_has_no_image_placeholder() -> None:
    value = _build_prompt_text(
        PromptProcessor(), "qwen2_5_vl", "Is it visible?", has_image=False
    )
    assert "Is it visible?" in value
    assert "image" not in value.lower()


def test_limit_preserves_all_jsonl_records(tmp_path: Path) -> None:
    samples_dir = tmp_path / "samples"
    samples_dir.mkdir()
    source = samples_dir / "input.jsonl"
    output = samples_dir / "output.jsonl"
    rows = [
        {
            "record_type": "run",
            "run": {
                "dataset": "test",
                "model_family": "llava_1_5",
                "model_runtime": {"engine": "vllm"},
                "prompt_sha256": "prompt",
            },
        },
        {"record_type": "sample", "sample": {"sample_id": "one"}, "samples": []},
        {"record_type": "sample", "sample": {"sample_id": "two"}, "samples": []},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert backfill_hidden_states(EmptyBackend(), {}, source, output, limit=1) == 0
    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(written) == len(rows)
    assert written[0]["backfill_manifest"]["version"] == "3.0"


def test_current_sidecar_rejects_old_source_and_bad_shape(tmp_path: Path) -> None:
    path = tmp_path / "hidden.pt"
    torch.save(torch.zeros((2, 4), dtype=torch.float16), path)
    item = {"hidden_states": {"source": "old", "shape": [2, 4]}}
    assert not _current_sidecar(item, path, 2)
    item["hidden_states"]["source"] = BACKFILL_SOURCE
    assert _current_sidecar(item, path, 2)
    assert not _current_sidecar(item, path, 3)
