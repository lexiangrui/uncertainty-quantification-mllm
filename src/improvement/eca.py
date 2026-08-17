"""ECA (Evidence-Chain Attention) — attention-based UQ.

Evidence-grounding uncertainty: does the information the final answer
relies on trace back to original visual evidence along the attention path
I → V → R → A?

Five token regions are distinguished:

    I  image tokens                (original visual evidence)
    Q  question tokens             (prompt text minus image)
    V  <vision>…</vision>          (self-generated visual description)
    R  <reasoning>…</reasoning>    (self-generated reasoning)
    A  <answer>…</answer>          (self-generated answer)

For every decoder layer and every source group S ∈ {V, R, A} we record the
attention mass from the group's prediction rows (row t-1 predicts token t,
the PAS convention) to every destination bucket in {I, Q, V, R, A}:

    M[l][S][T] = Σ_{rows t∈S} Σ_{heads h} Σ_{j∈T} A^(l,h)[t-1, j]

All derived scores (G_V, G_R, G_A, U_direct, U_ECA per layer) are computed
offline from these masses, so layer sweeps never need a re-run.

**Memory-efficient exact attention**: same chunked eager scheme as the
GCAR pipeline — swap ``ALL_ATTENTION_FUNCTIONS["eager"]`` for a chunked
implementation (post-RoPE Q/K, additive mask, fp32 softmax, adaptive
~512 MB tiles); rows are accumulated inside the chunk loop.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass

from src.improvement.backend import GcarBackend

_CHUNK = 256

GROUPS = ("vision", "reasoning", "answer")
DESTS = ("image", "question", "vision", "reasoning", "answer")


@dataclass
class EcaResult:
    n_visual_tokens: int
    n_heads: int
    section_tokens: dict[str, int]  # rows per group = section token count
    # {layer: [[mI, mQ, mV, mR, mA] per source group, in GROUPS order]}
    layer_masses: dict[int, list[list[float]]]

    def to_dict(self) -> dict:
        return {
            "n_visual_tokens": self.n_visual_tokens,
            "n_heads": self.n_heads,
            "section_tokens": self.section_tokens,
            "layer_masses": {str(k): v for k, v in self.layer_masses.items()},
        }


def _get_decoder_layers(backend: GcarBackend):
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


class _EcaAccumulator:
    """Accumulates group→bucket attention mass per decoder layer.

    predict_idx: sorted union of the prediction rows of all three groups;
    group_boundaries: row-count boundaries (exclusive) of vision/reasoning/
    answer within predict_idx; col_bucket: (seq,) destination bucket per
    position (-1 = unbucketed, e.g. tag gaps / BOS).
    """

    def __init__(
        self,
        predict_idx: torch.Tensor,
        group_boundaries: list[int],
        col_bucket: torch.Tensor,
    ):
        self.predict_idx = predict_idx.cpu()
        self.boundaries = torch.tensor(group_boundaries, dtype=torch.long)
        self.col_bucket = col_bucket.cpu()
        self.module_layers: dict[int, int] = {}
        # layer → [group][bucket] head-summed mass
        self.layer_masses: dict[int, list[list[float]]] = {}
        self.n_heads = 0
        self._onehot: torch.Tensor | None = None

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
        self.n_heads = probs.shape[1]

        # (n_sel, 5): head-summed mass of each selected row per bucket.
        col_rows = rows.sum(dim=0).float()  # (n_sel, seq)
        if self._onehot is None or self._onehot.shape[0] != probs.shape[3]:
            bucket = self.col_bucket.to(probs.device)
            self._onehot = torch.nn.functional.one_hot(
                bucket.clamp_min(0), num_classes=5
            ).float() * (bucket >= 0).unsqueeze(-1).float()
        masses = col_rows @ self._onehot  # (n_sel, 5)

        # Group label of each selected row = its index within the sorted
        # union of prediction rows (predict_idx is the concatenation of the
        # three groups' contiguous row ranges).
        row_base = int(self.predict_idx[0].item())
        idx = sel - row_base
        labels = torch.searchsorted(self.boundaries.to(probs.device), idx, right=True)
        entry = self.layer_masses.get(layer)
        if entry is None:
            entry = self.layer_masses[layer] = [[0.0] * 5 for _ in GROUPS]
        for g in range(len(GROUPS)):
            m = masses[labels == g]
            if m.numel():
                row = m.sum(dim=0).tolist()
                acc = entry[g]
                for b in range(5):
                    acc[b] += row[b]


def _make_chunked_eager(accumulator: _EcaAccumulator):
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


def compute_eca(
    backend: GcarBackend,
    full_inputs: dict,
    prompt_length: int,
    section_spans: dict[str, tuple[int, int]],
) -> EcaResult | None:
    """Compute per-layer evidence-chain attention masses."""
    input_ids = full_inputs["input_ids"]
    seq_len = int(input_ids.shape[1])
    image_token_id = backend._image_token_id

    v0, v1 = section_spans["vision"]
    r0, r1 = section_spans["reasoning"]
    a0, a1 = section_spans["answer"]
    spans = {"vision": (v0, v1), "reasoning": (r0, r1), "answer": (a0, a1)}
    if any(e - s < 1 for s, e in spans.values()):
        return None

    visual_mask = (input_ids[0] == image_token_id)
    n_visual = int(visual_mask.sum().item())
    if n_visual == 0:
        return None

    # Prediction rows per group: row t-1 predicts token t.
    predict_idx = torch.cat([torch.arange(s - 1, e - 1) for s, e in spans.values()])
    boundaries = [v1 - v0, (v1 - v0) + (r1 - r0), (v1 - v0) + (r1 - r0) + (a1 - a0)]

    # Destination bucket per position: 0 image, 1 question, 2 V, 3 R, 4 A;
    # tag-gap tokens between sections stay unbucketed (-1).
    col_bucket = torch.full((seq_len,), -1, dtype=torch.long)
    col_bucket[:prompt_length] = 1  # question side (image set below)
    col_bucket[visual_mask.cpu()] = 0
    col_bucket[v0:v1] = 2
    col_bucket[r0:r1] = 3
    col_bucket[a0:a1] = 4

    layers = _get_decoder_layers(backend)
    accumulator = _EcaAccumulator(predict_idx, boundaries, col_bucket)
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

    if not accumulator.layer_masses:
        return None
    return EcaResult(
        n_visual_tokens=n_visual,
        n_heads=accumulator.n_heads,
        section_tokens={name: spans[name][1] - spans[name][0] for name in GROUPS},
        layer_masses=accumulator.layer_masses,
    )
