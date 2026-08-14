"""Visual Grounding Score (VGS) — attention-based UQ.

Measures the fraction of attention from answer tokens directed at visual
tokens vs. text tokens.  Low visual grounding → answer generated from
language context → hallucination risk.

**Memory-efficient exact attention**: ``attn_implementation='eager'`` makes
transformers materialize a full (heads × seq × seq) matrix per layer —
~1.4 GB at seq≈3500, which OOMs for the longest InternVL3.5 sequences.
Instead we swap the registry entry ``ALL_ATTENTION_FUNCTIONS["eager"]``
for a *chunked* implementation: query rows are processed in blocks of
``_CHUNK`` so peak memory is one (heads × chunk × seq) tile, while the
math (post-RoPE Q/K, additive mask, fp32 softmax) is identical to eager.
The VGS rows (answer-predicting positions) are accumulated inside the
chunk loop and never stored in full.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass

from src.improvement.lac import LacBackend

_CHUNK = 256


@dataclass
class VgsResult:
    score: float | None
    vision_attn_ratio: float | None
    vision_attn_mean: float | None
    n_answer_tokens: int
    n_visual_tokens: int

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "vision_attn_ratio": self.vision_attn_ratio,
            "vision_attn_mean": self.vision_attn_ratio,
            "n_answer_tokens": self.n_answer_tokens,
            "n_visual_tokens": self.n_visual_tokens,
        }


def _get_decoder_layers(backend: LacBackend):
    base = backend._base_model()
    core = getattr(base, "model", base)
    lang_model = getattr(core, "language_model", core)
    decoder = getattr(lang_model, "model", lang_model)
    layers = getattr(decoder, "layers", None)
    if layers is not None:
        return list(layers)
    raise RuntimeError("cannot locate decoder layers")


def _repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 1:
        return x
    return x[:, :, None, :, :].expand(*x.shape[:2], n, *x.shape[2:]).reshape(
        x.shape[0], x.shape[1] * n, x.shape[2], x.shape[3]
    )


class _VgsAccumulator:
    """Accumulates answer→visual attention mass per selected decoder layer."""

    def __init__(self, predict_idx: torch.Tensor, visual_idx: torch.Tensor, selected_layers: set[int]):
        self.predict_idx = predict_idx.cpu()
        self.visual_idx = visual_idx.cpu()
        self.selected = selected_layers
        self.module_layers: dict[int, int] = {}
        self.visual_sum = 0.0
        self.total_sum = 0.0

    def map_modules(self, layers: list) -> None:
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                self.module_layers[id(attn)] = i

    def accumulate(self, module, probs: torch.Tensor, row_start: int) -> None:
        layer = self.module_layers.get(id(module))
        if layer is None or layer not in self.selected:
            return
        p = self.predict_idx.to(probs.device)
        sel = p[(p >= row_start) & (p < row_start + probs.shape[2])]
        if sel.numel() == 0:
            return
        rows = probs[0][:, sel - row_start, :]  # (heads, n_sel, seq)
        col = rows.sum(dim=(0, 1))  # (seq,)
        vidx = self.visual_idx.to(col.device)
        self.visual_sum += col[vidx].sum().item()
        self.total_sum += col.sum().item()


def _make_chunked_eager(accumulator: _VgsAccumulator):
    def chunked_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        is_causal = kwargs.get("is_causal")
        is_decoder = id(module) in accumulator.module_layers
        groups = getattr(module, "num_key_value_groups", 1)
        key_states = _repeat_kv(key, groups)
        value_states = _repeat_kv(value, groups)
        q_len = query.shape[2]
        kv_len = key_states.shape[2]

        # Adaptive chunk: keep one fp32 probs tile under ~512 MB even for
        # extreme sequences (Qwen2.5-VL global-attention ViT layers can see
        # tens of thousands of visual tokens on high-resolution images).
        budget = 512 * 1024 * 1024
        chunk = max(64, min(_CHUNK, int(budget / (query.shape[1] * kv_len * 4))))

        out = torch.empty_like(query)
        for s in range(0, q_len, chunk):
            e = min(s + chunk, q_len)
            scores = torch.matmul(query[:, :, s:e, :], key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, s:e, :]
            elif is_causal or (is_decoder and is_causal is None):
                # Decoder rows are causal unless an explicit mask was given.
                rows = torch.arange(s, e, device=scores.device).unsqueeze(1)
                cols = torch.arange(kv_len, device=scores.device).unsqueeze(0)
                causal = cols <= rows
                scores = scores.masked_fill(
                    ~causal.view(1, 1, e - s, kv_len), torch.finfo(scores.dtype).min
                )
            # else: bidirectional (ViT global attention) — no mask
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            out[:, :, s:e, :] = torch.matmul(probs, value_states)
            accumulator.accumulate(module, probs, s)
        return out.transpose(1, 2).contiguous(), None

    return chunked_eager


def compute_vgs(
    backend: LacBackend,
    full_inputs: dict,
    prompt_length: int,
    answer_span: tuple[int, int],
) -> VgsResult:
    """Compute Visual Grounding Score via chunked eager attention."""
    input_ids = full_inputs["input_ids"]
    image_token_id = backend._image_token_id
    ans_start, ans_end = answer_span
    n_ans = ans_end - ans_start
    if n_ans < 1:
        return VgsResult(None, None, None, 0, 0)

    visual_mask = (input_ids[0] == image_token_id)
    n_visual = visual_mask.sum().item()
    if n_visual == 0:
        return VgsResult(None, None, None, n_ans, 0)

    predict_idx = torch.arange(ans_start - 1, ans_end - 1)
    visual_idx = visual_mask.nonzero(as_tuple=True)[0].cpu()

    layers = _get_decoder_layers(backend)
    n_layers = len(layers)
    selected = set(range(n_layers // 3, n_layers))

    accumulator = _VgsAccumulator(predict_idx, visual_idx, selected)
    accumulator.map_modules(layers)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # "eager" is not in the default registry (models pass it as a default
    # argument), so get_interface() falls back to our entry once registered.
    ALL_ATTENTION_FUNCTIONS["eager"] = _make_chunked_eager(accumulator)
    try:
        with torch.inference_mode():
            outputs = backend.model(**full_inputs, use_cache=False)
    finally:
        del ALL_ATTENTION_FUNCTIONS["eager"]

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if accumulator.total_sum < 1e-12:
        return VgsResult(None, None, None, n_ans, n_visual)

    vision_ratio = accumulator.visual_sum / accumulator.total_sum
    return VgsResult(
        score=-vision_ratio,
        vision_attn_ratio=vision_ratio,
        vision_attn_mean=vision_ratio,
        n_answer_tokens=n_ans,
        n_visual_tokens=n_visual,
    )
