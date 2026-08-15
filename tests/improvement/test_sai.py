"""Unit tests for the SAI core: perturbation geometry, hook contract,
anchor construction and span mapping."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.improvement.sai import (
    InterventionHook,
    LensKit,
    anchor_token_id,
    renorm_rotation,
    section_spans,
    stable_seed,
    unit_direction,
)


# ----------------------------------------------------------------------
# Perturbation geometry
# ----------------------------------------------------------------------

def test_renorm_rotation_preserves_norms():
    torch.manual_seed(0)
    h = torch.randn(1, 5, 8)
    disp = torch.randn(1, 5, 8) * 0.3
    out = renorm_rotation(h, disp)
    assert torch.allclose(out.norm(dim=-1), h.norm(dim=-1), atol=1e-5)


def test_renorm_rotation_moves_toward_displacement():
    h = torch.zeros(1, 1, 8)
    h[0, 0, 0] = 1.0
    u = torch.zeros(8)
    u[1] = 1.0
    out = renorm_rotation(h, 1.0 * u)  # sigma=1 along y
    # cos = 1/sqrt(2) along x, sin = 1/sqrt(2) along y
    assert math.isclose(out[0, 0, 0].item(), 1 / math.sqrt(2), rel_tol=1e-5)
    assert math.isclose(out[0, 0, 1].item(), 1 / math.sqrt(2), rel_tol=1e-5)


def test_unit_direction_deterministic():
    a = unit_direction(16, stable_seed("x", 1), "cpu")
    b = unit_direction(16, stable_seed("x", 1), "cpu")
    c = unit_direction(16, stable_seed("x", 2), "cpu")
    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)
    assert math.isclose(float(a.norm()), 1.0, rel_tol=1e-6)


# ----------------------------------------------------------------------
# Intervention hook contract (transformers >= 5 bare tensor, older tuple)
# ----------------------------------------------------------------------

class _Layer:
    def forward(self, hidden_states):
        return hidden_states


def _run_hook(hook, hidden, args_form):
    layer = _Layer()
    if args_form == "kwargs":
        args, kwargs = (), {"hidden_states": hidden}
    elif args_form == "tuple":
        args, kwargs = (hidden,), {}
    else:
        args, kwargs = (hidden,), {"something": 1}
    out = hook(layer, args, kwargs)
    assert isinstance(out, tuple) and len(out) == 2
    new_args, new_kwargs = out
    if args_form == "kwargs":
        return new_kwargs["hidden_states"]
    return new_args[0]


@pytest.mark.parametrize("form", ["kwargs", "tuple", "mixed"])
def test_hook_returns_args_kwargs_contract(form):
    torch.manual_seed(1)
    h = torch.randn(1, 4, 8)
    hook = InterventionHook()
    hook.arm(positions=torch.tensor([1, 3]), direction=unit_direction(8, 7, "cpu"), sigma=1.0)
    out = _run_hook(hook, h.clone(), form)
    untouched = [0, 2]
    assert torch.allclose(out[0, untouched], h[0, untouched])
    assert not torch.allclose(out[0, 1, :], h[0, 1, :])
    assert torch.allclose(out.norm(dim=-1), h.norm(dim=-1), atol=1e-5)


def test_hook_per_position_direction_and_weights():
    torch.manual_seed(2)
    h = torch.randn(1, 3, 8)
    d = torch.randn(2, 8)  # positions 1 and 2
    w = torch.tensor([1.0, 0.0])
    hook = InterventionHook()
    hook.arm(positions=torch.tensor([1, 2]), direction=d, sigma=2.0, weights=w)
    out = _run_hook(hook, h.clone(), "kwargs")
    # weight 0 → unchanged direction magnitude 0 → token unchanged
    assert torch.allclose(out[0, 2], h[0, 2], atol=1e-6)
    assert not torch.allclose(out[0, 1], h[0, 1])


def test_hook_disarmed_passthrough():
    hook = InterventionHook()
    assert hook(object(), (), {}) is None


# ----------------------------------------------------------------------
# Anchor construction
# ----------------------------------------------------------------------

class _FakeNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x


class _FakeBackend:
    class _M:
        lm_head = None

    def __init__(self, dim=6, vocab=4):
        self._W = torch.eye(dim)[:vocab] if vocab <= dim else torch.randn(vocab, dim)
        self._norm = _FakeNorm(dim)
        self._head = torch.nn.Linear(dim, vocab, bias=False)
        with torch.no_grad():
            self._head.weight.copy_(self._W)
        self._m = self._M()
        self._m.lm_head = self._head

    def _base_model(self):
        return self._m

    def lm_head_weight(self):
        return self._W

    def final_norm(self):
        return self._norm


def test_anchor_direction_centered_and_orthogonal_to_mean():
    backend = _FakeBackend(dim=6, vocab=4)
    kit = LensKit(backend)
    u = kit.anchor_direction(1, center=True)
    assert math.isclose(float(u.norm()), 1.0, rel_tol=1e-5)
    # centered: orthogonal-ish to the mean row direction in post-norm space
    r = backend.lm_head_weight()[1] - backend.lm_head_weight().float().mean(0)
    cos = float(torch.nn.functional.cosine_similarity(u, r, dim=0))
    assert cos > 0.99


def test_anchor_direction_uncentered_aligns_with_row():
    backend = _FakeBackend(dim=6, vocab=4)
    kit = LensKit(backend)
    u = kit.anchor_direction(2, center=False)
    cos = float(torch.nn.functional.cosine_similarity(
        u, backend.lm_head_weight()[2].float(), dim=0))
    assert cos > 0.99


# ----------------------------------------------------------------------
# Object → token id and span mapping
# ----------------------------------------------------------------------

class _FakeTok:
    all_special_ids = [0]

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text if c != ""]

    def decode(self, ids, clean_up_tokenization_spaces=True):
        return "".join(chr(i) for i in ids)


class _FakeTokBackend:
    tokenizer = _FakeTok()


def test_anchor_token_id_skips_whitespace_tokens():
    # " dog" encodes as [' ', 'd', 'o', 'g']; the standalone space (ord 32,
    # which decodes to whitespace-only) must be skipped.
    assert anchor_token_id(_FakeTokBackend(), "dog") == ord("d")


def test_section_spans_and_mapping():
    raw = "<vision>A dog.</vision><reasoning>It is a dog.</reasoning><answer>dog</answer>"
    spans = section_spans(raw)
    assert raw[spans.vision[0]:spans.vision[1]] == "A dog."
    assert raw[spans.answer[0]:spans.answer[1]] == "dog"


def test_char_span_tokens_offset_mapping():
    from src.improvement.backend import TeacherForcingBatch

    batch = TeacherForcingBatch(
        inputs={}, prompt_length=10, gen_ids=[1, 2, 3],
        offsets=[(0, 2), (2, 5), (5, 6)],
    )
    assert batch.char_span_tokens(0, 2) == [0]
    assert batch.char_span_tokens(2, 5) == [1]
    assert batch.char_span_tokens(0, 6) == [0, 1, 2]
    assert batch.char_span_tokens(9, 12) == []
    assert batch.absolute(2) == 12
