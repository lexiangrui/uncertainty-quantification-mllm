"""EPAR (Early Prelim Attention Ratio) — attention-based UQ.

Measures the fraction of answer-generation attention routed to the model's
own already-generated text (prelim: reasoning scaffolding + answer prefix,
excluding the row's own position) in the FIRST FOUR decoder layers.  High
self-routing → the answer is generated from self-reinforcement rather than
visual evidence → hallucination risk.

Each answer row (position p, predicting the next answer token) has its
causal context [0, p] split into five disjoint regions — visual / scaffold /
prefix / self / prompt_text — and all five masses are recorded for every
decoder layer, so any score variant is recoverable offline
(see scripts/analysis/ablate_epar.py).

**Memory-efficient exact attention**: ``attn_implementation='eager'`` makes
transformers materialize a full (heads × seq × seq) matrix per layer —
~1.4 GB at seq≈3500, which OOMs for the longest InternVL3.5 sequences.
Instead we swap the registry entry ``ALL_ATTENTION_FUNCTIONS["eager"]``
for a *chunked* implementation: query rows are processed in blocks of
``_CHUNK`` so peak memory is one (heads × chunk × seq) tile, while the
math (post-RoPE Q/K, additive mask, fp32 softmax) is identical to eager.
The answer rows are accumulated inside the chunk loop and never stored in
full.
"""
from __future__ import annotations

import math

import torch
from dataclasses import dataclass

from src.improvement.backend import EparBackend

_CHUNK = 256

# EPAR aggregation window: the first four decoder layers, frozen on the dev
# model (llava); insensitive between 1 and 4 layers (see ablation).
_EPAR_LAYERS = 4


@dataclass
class EparResult:
    # EPAR: prelim mass fraction over the first four decoder layers.
    score: float | None
    prelim_attn_ratio: float | None  # identical to score (explicit name)
    # Diagnostic: visual mass fraction over the selected (last-2/3) layers.
    vision_attn_ratio: float | None
    n_answer_tokens: int
    n_visual_tokens: int
    # Region masses over the selected (last-2/3) layers.
    visual_attn_sum: float | None = None
    prelim_attn_sum: float | None = None  # scaffold + prefix + self
    prompt_text_attn_sum: float | None = None
    # Mass-weighted normalized entropy of the answer→visual attention
    # distribution (1 = uniform over visual tokens, 0 = single patch).
    visual_attn_entropy: float | None = None
    # Per-layer region sums for every decoder layer:
    # {layer: [vis, scaffold, prefix, self, text]}
    layer_breakdown: dict[int, list[float]] | None = None
    # Per-layer dispersion accumulators: {layer: [h_sum, s_sum]}; dispersion
    # over a layer range = sum(h)/sum(s).
    layer_dispersion: dict[int, list[float]] | None = None

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "prelim_attn_ratio": self.prelim_attn_ratio,
            "vision_attn_ratio": self.vision_attn_ratio,
            "n_answer_tokens": self.n_answer_tokens,
            "n_visual_tokens": self.n_visual_tokens,
            "visual_attn_sum": self.visual_attn_sum,
            "prelim_attn_sum": self.prelim_attn_sum,
            "prompt_text_attn_sum": self.prompt_text_attn_sum,
            "visual_attn_entropy": self.visual_attn_entropy,
            "layer_breakdown": self.layer_breakdown,
            "layer_dispersion": self.layer_dispersion,
        }


def _get_decoder_layers(backend: EparBackend):
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


class _EparAccumulator:
    """Accumulates answer→region attention mass for every decoder layer.

    For every answer row (position p predicts the next answer token) the
    causal context [0, p] is partitioned into five disjoint regions:

    - visual: image token positions
    - scaffold: generated tokens before the answer span [prompt_length, ans_start-1]
    - prefix: answer tokens strictly before the row's own position [ans_start, p-1]
    - self: the row's own position {p} (self-attention; excluded from prelim)
    - prompt_text: prompt positions [0, prompt_length) minus visual

    visual + scaffold + prefix + self + prompt_text equals the full context
    of every row; prelim = scaffold + prefix.
    """

    def __init__(
        self,
        predict_idx: torch.Tensor,
        visual_idx: torch.Tensor,
        selected_layers: set[int],
        prompt_length: int,
    ):
        self.predict_idx = predict_idx.cpu()
        self.visual_idx = visual_idx.cpu()
        self.prompt_length = prompt_length
        self.selected = selected_layers
        self.module_layers: dict[int, int] = {}
        self.visual_sum = 0.0
        self.total_sum = 0.0
        self.prelim_sum = 0.0
        self.prompt_text_sum = 0.0
        # answer span start in absolute positions (predict rows are
        # [ans_start-1, ans_end-1])
        self.ans_start = int(self.predict_idx[0].item()) + 1
        # Per-layer region sums [vis, scaffold, prefix, self, text] for every
        # decoder layer, so any layer range can be reconstructed offline.
        self.layer_breakdown: dict[int, list[float]] = {}
        # Per-layer dispersion accumulators [weighted entropy sum, visual
        # mass]; dispersion over a layer range = sum(h)/sum(s) over the range.
        self.layer_dispersion: dict[int, list[float]] = {}

    def map_modules(self, layers: list) -> None:
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                self.module_layers[id(attn)] = i

    def accumulate(self, module, probs: torch.Tensor, row_start: int) -> None:
        layer = self.module_layers.get(id(module))
        if layer is None:
            return
        p = self.predict_idx.to(probs.device)
        sel = p[(p >= row_start) & (p < row_start + probs.shape[2])]
        if sel.numel() == 0:
            return
        rows = probs[0][:, sel - row_start, :]  # (heads, n_sel, seq)
        vidx = self.visual_idx.to(rows.device)

        # Region masses. col_rows[k, j] = attention of the k-th selected row
        # to position j, summed over heads.
        col_rows = rows.sum(dim=0).float()  # (n_sel, seq)
        vis_rows = col_rows[:, vidx].sum(dim=-1)  # (n_sel,)
        pl = self.prompt_length
        as_ = self.ans_start
        parts = []
        for k, p_k in enumerate(sel):
            pos = int(p_k.item())
            scaffold = col_rows[k, pl : min(pos, as_ - 1) + 1].sum()
            prefix = col_rows[k, as_ : pos].sum()
            self_mass = col_rows[k, pos].sum()
            parts.append((scaffold, prefix, self_mass))
        scaffold_rows = torch.stack([a for a, _, _ in parts])
        prefix_rows = torch.stack([b for _, b, _ in parts])
        self_rows = torch.stack([c for _, _, c in parts])
        text_rows = col_rows[:, :pl].sum(dim=-1) - vis_rows

        vis = vis_rows.sum().item()
        sca = scaffold_rows.sum().item()
        pre = prefix_rows.sum().item()
        sel_ = self_rows.sum().item()
        txt = text_rows.sum().item()
        b = self.layer_breakdown.get(layer)
        if b is None:
            b = self.layer_breakdown[layer] = [0.0, 0.0, 0.0, 0.0, 0.0]
        b[0] += vis
        b[1] += sca
        b[2] += pre
        b[3] += sel_
        b[4] += txt
        if layer in self.selected:
            self.visual_sum += vis
            self.prelim_sum += sca + pre + sel_
            self.prompt_text_sum += txt
            self.total_sum += vis + sca + pre + sel_ + txt

        # Dispersion: normalized entropy of the per-(head, row) visual
        # attention distribution, renormalized over visual positions.
        n_vis = vidx.numel()
        if n_vis > 1:
            p_v = rows[:, :, vidx].float()  # (heads, n_sel, n_vis)
            s = p_v.sum(dim=-1)  # (heads, n_sel)
            e = (p_v * torch.log(p_v + 1e-12)).sum(dim=-1)
            s_cl = s.clamp_min(1e-12)
            h = (torch.log(s_cl) - e / s_cl) / math.log(n_vis)
            d = self.layer_dispersion.get(layer)
            if d is None:
                d = self.layer_dispersion[layer] = [0.0, 0.0]
            d[0] += (h * s).sum().item()
            d[1] += s.sum().item()

    def dispersion(self, layers: set[int] | None = None) -> float | None:
        """Mass-weighted normalized entropy over `layers` (default: selected)."""
        sel = self.selected if layers is None else layers
        h_sum = sum(v[0] for k, v in self.layer_dispersion.items() if k in sel)
        s_sum = sum(v[1] for k, v in self.layer_dispersion.items() if k in sel)
        if s_sum < 1e-12:
            return None
        return float(h_sum / s_sum)

    def epar(self) -> float | None:
        """Prelim (scaffold+prefix) fraction over the first _EPAR_LAYERS layers."""
        rows = [self.layer_breakdown.get(i) for i in range(_EPAR_LAYERS)]
        if not all(rows):
            return None
        num = sum(r[1] + r[2] for r in rows)
        tot = sum(sum(r) for r in rows)
        if tot < 1e-12:
            return None
        return num / tot


def _make_chunked_eager(accumulator: _EparAccumulator):
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


def compute_epar(
    backend: EparBackend,
    full_inputs: dict,
    prompt_length: int,
    answer_span: tuple[int, int],
) -> EparResult:
    """Compute EPAR via chunked eager attention."""
    input_ids = full_inputs["input_ids"]
    image_token_id = backend._image_token_id
    ans_start, ans_end = answer_span
    n_ans = ans_end - ans_start
    if n_ans < 1:
        return EparResult(None, None, None, 0, 0)

    visual_mask = (input_ids[0] == image_token_id)
    n_visual = visual_mask.sum().item()
    if n_visual == 0:
        return EparResult(None, None, None, n_ans, 0)

    predict_idx = torch.arange(ans_start - 1, ans_end - 1)
    visual_idx = visual_mask.nonzero(as_tuple=True)[0].cpu()

    layers = _get_decoder_layers(backend)
    n_layers = len(layers)
    selected = set(range(n_layers // 3, n_layers))

    accumulator = _EparAccumulator(predict_idx, visual_idx, selected, prompt_length)
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
        return EparResult(None, None, None, n_ans, n_visual)

    epar = accumulator.epar()
    vision_ratio = accumulator.visual_sum / accumulator.total_sum
    return EparResult(
        score=epar,
        prelim_attn_ratio=epar,
        vision_attn_ratio=vision_ratio,
        n_answer_tokens=n_ans,
        n_visual_tokens=n_visual,
        visual_attn_sum=accumulator.visual_sum,
        prelim_attn_sum=accumulator.prelim_sum,
        prompt_text_attn_sum=accumulator.prompt_text_sum,
        visual_attn_entropy=accumulator.dispersion(),
        layer_breakdown=accumulator.layer_breakdown,
        layer_dispersion=accumulator.layer_dispersion,
    )
