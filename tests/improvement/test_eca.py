from pathlib import Path

import pytest
import torch

from src.improvement.backend import EcaBackend
from src.improvement.eca import _EcaAccumulator, layer_features


class _FastTokenizer:
    is_fast = True

    def __init__(self, offsets):
        self.offsets = offsets

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": list(range(len(self.offsets))),
            "offset_mapping": self.offsets,
        }


class _Processor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def test_section_token_spans_exclude_tokens_crossing_xml_boundaries():
    backend = EcaBackend("llava_1_5", Path("unused"))
    backend.processor = _Processor(
        _FastTokenizer(
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
            ]
        )
    )

    ids, spans = backend._section_token_spans(
        "x" * 79,
        {"vision": (8, 18), "reasoning": (40, 45), "answer": (64, 69)},
    )

    assert ids == list(range(9))
    assert spans == {"vision": (1, 2), "reasoning": (4, 5), "answer": (7, 8)}


def test_section_token_spans_require_fast_tokenizer():
    backend = EcaBackend("llava_1_5", Path("unused"))
    tokenizer = _FastTokenizer([])
    tokenizer.is_fast = False
    backend.processor = _Processor(tokenizer)

    with pytest.raises(ValueError, match="fast tokenizer"):
        backend._section_token_spans("text", {"vision": (0, 4)})


def test_accumulator_preserves_groups_across_xml_gaps():
    module = object()
    accumulator = _EcaAccumulator(
        predict_idx=torch.tensor([10, 11, 20, 21, 30]),
        row_groups=torch.tensor([0, 0, 1, 1, 2]),
        col_bucket=torch.tensor([0, -1, -1, -1]),
    )
    accumulator.module_layers[id(module)] = 0

    for row_start, rows in ((10, 2), (20, 2), (30, 1)):
        probs = torch.zeros((1, 1, rows, 4))
        probs[..., 0] = 1.0
        accumulator.accumulate(module, probs, row_start)

    assert accumulator.layer_masses[0] == [
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ]


def test_layer_features_include_same_section_attention_in_denominators():
    result = {
        "n_heads": 1,
        "section_tokens": {"vision": 1, "reasoning": 1, "answer": 1},
        "layer_masses": {
            "0": [
                [1.0, 0.0, 9.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 9.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 9.0],
            ]
        },
    }

    features = layer_features(result)[0]

    assert features == pytest.approx(
        {"U_image": 0.0, "U_direct": 0.9, "U_V": 0.9, "U_R": 0.9, "U_ECA": 0.9}
    )
