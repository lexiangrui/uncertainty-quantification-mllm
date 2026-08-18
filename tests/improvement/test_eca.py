from pathlib import Path

import pytest
import torch

from src.improvement.backend import EcaBackend
from src.improvement.eca import _EcaAccumulator, layer_features


class _MockTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self, token_map: dict[tuple[int, ...], str]):
        self.token_map = token_map

    def decode(self, token_ids, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return self.token_map.get(tuple(token_ids), "")


class _Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def _backend(tokenizer):
    backend = EcaBackend("llava_1_5", Path("unused"))
    backend.processor = _Processor(tokenizer)
    return backend


def test_continuous_slice_token_alignment():
    # Token stream representing "<vision>see</vision><reasoning>think</reasoning><answer>4</answer>"
    token_ids = [10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 35]
    # token 10-15: vision (<vision>see</vision>)
    # token 20-25: reasoning (<reasoning>think</reasoning>)
    # token 30-35: answer (<answer>4</answer>)
    token_map = {}
    for i in range(1, len(token_ids) + 1):
        sub = token_ids[:i]
        if i <= 6:
            text = "<vision>see</vision>"[: i * 3]
        elif i <= 12:
            text = "<vision>see</vision><reasoning>think</reasoning>"[: 20 + (i - 6) * 4]
        else:
            text = "<vision>see</vision><reasoning>think</reasoning><answer>4</answer>"[: 48 + (i - 12) * 3]
        token_map[tuple(sub)] = text

    tok = _MockTokenizer(token_map)
    backend = _backend(tok)

    ids, buckets = backend._align_generated_tokens(
        "<vision>see</vision><reasoning>think</reasoning><answer>4</answer>",
        token_ids + [tok.eos_token_id],
    )

    assert ids == token_ids
    assert len(buckets) == len(token_ids)
    # Tokens 0..5 in bucket 2 (vision), 6..11 in bucket 3 (reasoning), 12..17 in bucket 4 (answer)
    assert buckets[:6] == [2] * 6
    assert buckets[6:12] == [3] * 6
    assert buckets[12:] == [4] * 6


def test_alignment_rejects_missing_section():
    # Only vision and reasoning, missing answer
    token_map = {
        (1,): "<vision>",
        (1, 2): "<vision>v</vision>",
        (1, 2, 3): "<vision>v</vision><reasoning>",
        (1, 2, 3, 4): "<vision>v</vision><reasoning>r</reasoning>",
    }
    tok = _MockTokenizer(token_map)
    with pytest.raises(ValueError, match="does not contain valid"):
        _backend(tok)._align_generated_tokens("raw_missing_answer", [1, 2, 3, 4])


def test_accumulator_preserves_groups_and_five_buckets():
    module = object()
    accumulator = _EcaAccumulator(
        predict_idx=torch.tensor([10, 11, 20, 21, 30]),
        row_groups=torch.tensor([0, 0, 1, 1, 2]),
        col_bucket=torch.tensor([0, 1, 2, 3, 4]),
    )
    accumulator.module_layers[id(module)] = 0

    for row_start, rows in ((10, 2), (20, 2), (30, 1)):
        probs = torch.zeros((1, 1, rows, 5))
        probs[..., 0] = 1.0
        accumulator.accumulate(module, probs, row_start)

    assert accumulator.layer_masses[0] == [
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_accumulator_rejects_invalid_bucket():
    with pytest.raises(ValueError, match=r"\[0, 4\]"):
        _EcaAccumulator(
            predict_idx=torch.tensor([0]),
            row_groups=torch.tensor([0]),
            col_bucket=torch.tensor([-1]),
        )


def test_layer_features_computes_u_direct_and_u_direct_no_aa():
    # masses for answer rows: aAI=1.0, aAQ=2.0, aAV=3.0, aAR=4.0, aAA=5.0
    # heads=1, n_answer=1
    result = {
        "n_heads": 1,
        "section_tokens": {"vision": 1, "reasoning": 1, "answer": 1},
        "layer_masses": {
            "0": [
                [0.0] * 5,
                [0.0] * 5,
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ]
        },
    }

    feats = layer_features(result)[0]
    # U_direct: (3.0 + 4.0) / (1.0 + 2.0 + 3.0 + 4.0 + 5.0) = 7.0 / 15.0
    assert feats["U_direct"] == pytest.approx(7.0 / 15.0)
    # U_direct_no_aa: (3.0 + 4.0) / (1.0 + 2.0 + 3.0 + 4.0) = 7.0 / 10.0 = 0.70
    assert feats["U_direct_no_aa"] == pytest.approx(7.0 / 10.0)


def test_get_decoder_layers_handles_various_architectures():
    from types import SimpleNamespace
    from src.improvement.eca import _get_decoder_layers

    # Case 1: base.model.layers
    b1 = SimpleNamespace(_base_model=lambda: SimpleNamespace(model=SimpleNamespace(layers=[1, 2, 3])))
    assert _get_decoder_layers(b1) == [1, 2, 3]

    # Case 2: base.language_model.model.layers
    b2 = SimpleNamespace(_base_model=lambda: SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=[4, 5]))))
    assert _get_decoder_layers(b2) == [4, 5]

    # Case 3: base.layers
    b3 = SimpleNamespace(_base_model=lambda: SimpleNamespace(layers=[6]))
    assert _get_decoder_layers(b3) == [6]


def test_prepare_inputs_sections_pops_position_ids_and_pads():
    token_map = {
        (1,): "<vision>",
        (1, 2): "<vision>v</vision><reasoning>",
        (1, 2, 3): "<vision>v</vision><reasoning>r</reasoning><answer>a</answer>",
    }
    backend = _backend(_MockTokenizer(token_map))
    backend.device = torch.device("cpu")
    backend._image_token_id = 0
    backend._prepare_prompt = lambda img, q: {
        "input_ids": torch.tensor([[10, 20]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "position_ids": torch.tensor([[[0, 1]]]),
        "mm_token_type_ids": torch.tensor([[0, 0]]),
    }

    raw = "<vision>v</vision><reasoning>r</reasoning><answer>a</answer>"
    full_inputs, prompt_length, buckets = backend.prepare_inputs_sections(
        None, "question", raw, [1, 2, 3]
    )

    assert prompt_length == 2
    assert "position_ids" not in full_inputs
    assert full_inputs["input_ids"].shape == (1, 5)
    assert full_inputs["attention_mask"].shape == (1, 5)
    assert full_inputs["mm_token_type_ids"].shape == (1, 5)
    assert full_inputs["mm_token_type_ids"].tolist() == [[0, 0, 0, 0, 0]]
