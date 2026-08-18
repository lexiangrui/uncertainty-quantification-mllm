from pathlib import Path

import pytest
import torch

from src.improvement.backend import EcaBackend
from src.improvement.eca import _EcaAccumulator, layer_features


class _FastTokenizer:
    is_fast = True
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self, text, input_ids, offsets):
        self.text = text
        self.input_ids = input_ids
        self.offsets = offsets

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return self.text

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert text == self.text
        assert not add_special_tokens
        assert return_offsets_mapping
        return {"input_ids": self.input_ids, "offset_mapping": self.offsets}


class _Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def _backend(tokenizer):
    backend = EcaBackend("llava_1_5", Path("unused"))
    backend.processor = _Processor(tokenizer)
    return backend


def test_exact_generated_ids_are_preserved_and_boundary_tokens_become_tags():
    text = "x" * 79
    token_ids = list(range(1, 10))
    tokenizer = _FastTokenizer(
        text,
        token_ids,
        [
            (0, 8),
            (8, 13),
            (13, 19),
            (19, 40),
            (40, 44),
            (44, 46),
            (46, 64),
            (64, 69),
            (69, 79),
        ],
    )

    ids, buckets = _backend(tokenizer)._align_generated_tokens(
        text,
        token_ids + [tokenizer.eos_token_id],
        {"vision": (8, 18), "reasoning": (40, 45), "answer": (64, 69)},
    )

    assert ids == token_ids
    assert buckets == [5, 2, 5, 5, 3, 5, 5, 4, 5]


def test_alignment_requires_fast_tokenizer():
    tokenizer = _FastTokenizer("text", [1], [(0, 4)])
    tokenizer.is_fast = False

    with pytest.raises(ValueError, match="fast tokenizer"):
        _backend(tokenizer)._align_generated_tokens(
            "text", [1], {"vision": (0, 4), "reasoning": (0, 4), "answer": (0, 4)}
        )


def test_alignment_rejects_ids_changed_by_full_response_encoding():
    tokenizer = _FastTokenizer("text", [1], [(0, 4)])

    with pytest.raises(ValueError, match="stable under full-response encoding"):
        _backend(tokenizer)._align_generated_tokens(
            "text", [2], {"vision": (0, 4), "reasoning": (0, 4), "answer": (0, 4)}
        )


def test_alignment_rejects_section_without_xml_free_token():
    tokenizer = _FastTokenizer(
        "aaaabbbbcccc<", [1, 2, 3], [(0, 4), (4, 8), (8, 13)]
    )

    with pytest.raises(ValueError, match="answer has no XML-free generated token"):
        _backend(tokenizer)._align_generated_tokens(
            "aaaabbbbcccc<",
            [1, 2, 3],
            {"vision": (0, 4), "reasoning": (4, 8), "answer": (8, 12)},
        )


def test_accumulator_preserves_groups_and_six_buckets():
    module = object()
    accumulator = _EcaAccumulator(
        predict_idx=torch.tensor([10, 11, 20, 21, 30]),
        row_groups=torch.tensor([0, 0, 1, 1, 2]),
        col_bucket=torch.tensor([0, 5, 5, 5]),
    )
    accumulator.module_layers[id(module)] = 0

    for row_start, rows in ((10, 2), (20, 2), (30, 1)):
        probs = torch.zeros((1, 1, rows, 4))
        probs[..., 0] = 1.0
        accumulator.accumulate(module, probs, row_start)

    assert accumulator.layer_masses[0] == [
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_accumulator_rejects_invalid_bucket():
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        _EcaAccumulator(
            predict_idx=torch.tensor([0]),
            row_groups=torch.tensor([0]),
            col_bucket=torch.tensor([-1]),
        )


def test_layer_features_returns_only_direct_attention_ratio():
    result = {
        "n_heads": 1,
        "section_tokens": {"vision": 1, "reasoning": 1, "answer": 1},
        "layer_masses": {
            "0": [
                [0.0] * 6,
                [0.0] * 6,
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        },
    }

    assert layer_features(result)[0] == pytest.approx({"U_direct": 7.0 / 15.0})
