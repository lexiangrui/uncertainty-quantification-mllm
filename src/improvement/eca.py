"""ECA (Evidence-Chain Attention) — attention-based UQ.

Evidence-grounding uncertainty: does the information the final answer
relies on trace back to original visual evidence along the attention path
I → V → R → A?

Five token regions are distinguished:

    I  image tokens                (original visual evidence)
    Q  prompt-text tokens          (system/question/control, minus image)
    V  <vision>…</vision>          (self-generated visual description)
    R  <reasoning>…</reasoning>    (self-generated reasoning)
    A  <answer>…</answer>          (self-generated answer)

For every decoder layer and every source group S ∈ {V, R, A} we record the
    attention mass from the group's prediction rows (row t-1 predicts token t)
    to every destination bucket in {I, Q, V, R, A}:

    M[l][S][T] = Σ_{rows t∈S} Σ_{heads h} Σ_{j∈T} A^(l,h)[t-1, j]

All derived scores (G_V, G_R, G_A, U_direct, U_ECA per layer) are computed
offline from these masses, so layer sweeps never need a re-run.

**Memory-efficient exact attention**: swap
``ALL_ATTENTION_FUNCTIONS["eager"]`` for a chunked implementation
(post-RoPE Q/K, additive mask, fp32 softmax, adaptive ~512 MB tiles);
rows are accumulated inside the chunk loop.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass

from src.improvement.backend import EcaBackend

_CHUNK = 256

GROUPS = ("vision", "reasoning", "answer")
DESTS = ("image", "prompt_text", "vision", "reasoning", "answer")
FEATURES = ("U_direct", "U_direct_no_aa")
EPS = 1e-8


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


def _get_decoder_layers(backend: EcaBackend):
    base = backend._base_model()
    candidates = [
        getattr(base, "language_model", None),
        getattr(getattr(base, "model", None), "language_model", None),
        getattr(base, "model", None),
        base,
    ]
    for c in candidates:
        if c is None:
            continue
        for target in (c, getattr(c, "model", None)):
            if target is not None:
                layers = getattr(target, "layers", None)
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
    row_groups: source-group label aligned one-to-one with predict_idx;
    col_bucket: destination bucket per sequence position (0..4).
    """

    def __init__(
        self,
        predict_idx: torch.Tensor,
        row_groups: torch.Tensor,
        col_bucket: torch.Tensor,
    ):
        self.predict_idx = predict_idx.cpu()
        self.row_groups = row_groups.cpu()
        self.col_bucket = col_bucket.cpu()
        if self.col_bucket.ndim != 1 or torch.any(
            (self.col_bucket < 0) | (self.col_bucket >= len(DESTS))
        ):
            raise ValueError(f"col_bucket values must be in [0, {len(DESTS) - 1}]")
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
        if (
            self._onehot is None
            or self._onehot.shape[0] != probs.shape[3]
            or self._onehot.device != probs.device
        ):
            bucket = self.col_bucket.to(probs.device)
            self._onehot = torch.nn.functional.one_hot(bucket, num_classes=len(DESTS)).float()
        masses = col_rows @ self._onehot  # (n_sel, 5)

        mask = (p >= row_start) & (p < row_start + probs.shape[2])
        labels = self.row_groups.to(probs.device)[mask]
        entry = self.layer_masses.get(layer)
        if entry is None:
            entry = self.layer_masses[layer] = [[0.0] * len(DESTS) for _ in GROUPS]
        for g in range(len(GROUPS)):
            m = masses[labels == g]
            if m.numel():
                row = m.sum(dim=0).tolist()
                acc = entry[g]
                for b in range(len(DESTS)):
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
        bytes_per_row = query.shape[1] * kv_len * 4
        if bytes_per_row > budget:
            raise RuntimeError(
                "one fp32 attention-probability row exceeds the 512 MiB budget"
            )
        chunk = max(1, min(_CHUNK, budget // bytes_per_row))

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
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
            out[:, :, s:e, :] = torch.matmul(probs.to(value_states.dtype), value_states)
            accumulator.accumulate(module, probs, s)
        return out.transpose(1, 2).contiguous(), None

    return chunked_eager


def compute_eca(
    backend: EcaBackend,
    full_inputs: dict,
    prompt_length: int,
    generated_buckets: list[int],
) -> EcaResult | None:
    """Compute per-layer direct-attention masses."""
    input_ids = full_inputs["input_ids"]
    seq_len = int(input_ids.shape[1])
    image_token_id = backend._image_token_id
    if image_token_id is None:
        return None

    if len(generated_buckets) != seq_len - prompt_length:
        raise ValueError("generated bucket count does not match the response length")
    generated_bucket_tensor = torch.tensor(generated_buckets, dtype=torch.long)
    positions = {
        name: torch.nonzero(generated_bucket_tensor == bucket, as_tuple=False).flatten()
        + prompt_length
        for name, bucket in zip(GROUPS, (2, 3, 4), strict=True)
    }
    if any(index.numel() == 0 for index in positions.values()):
        return None

    visual_mask = (input_ids[0] == image_token_id)
    n_visual = int(visual_mask.sum().item())
    if n_visual == 0:
        return None

    # Prediction rows per group: row t-1 predicts token t.
    predict_idx = torch.cat([positions[name] - 1 for name in GROUPS])
    row_groups = torch.cat([
        torch.full((positions[name].numel(),), group, dtype=torch.long)
        for group, name in enumerate(GROUPS)
    ])

    # Destination bucket per position: 0 image, 1 question, 2 V, 3 R, 4 A.
    col_bucket = torch.full((seq_len,), 1, dtype=torch.long)
    col_bucket[visual_mask.cpu()] = 0
    col_bucket[prompt_length:] = generated_bucket_tensor

    layers = _get_decoder_layers(backend)
    accumulator = _EcaAccumulator(predict_idx, row_groups, col_bucket)
    accumulator.map_modules(layers)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # "eager" is not in the default registry (models pass it as a default
    # argument), so get_interface() falls back to our entry once registered.
    previous_eager = ALL_ATTENTION_FUNCTIONS.get("eager")
    ALL_ATTENTION_FUNCTIONS["eager"] = _make_chunked_eager(accumulator)
    try:
        with torch.inference_mode():
            outputs = backend.model(**full_inputs, use_cache=False)
    finally:
        if previous_eager is None:
            ALL_ATTENTION_FUNCTIONS.pop("eager", None)
        else:
            ALL_ATTENTION_FUNCTIONS["eager"] = previous_eager

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not accumulator.layer_masses:
        return None
    return EcaResult(
        n_visual_tokens=n_visual,
        n_heads=accumulator.n_heads,
        section_tokens={name: positions[name].numel() for name in GROUPS},
        layer_masses=accumulator.layer_masses,
    )


# Decoder layers the method averages over (frozen 2026-08-18).
DIRECT_LAYERS = (0, 1)


def layer_features(result: dict) -> dict[int, dict[str, float]]:
    """Per-layer U_direct (standard) and U_direct_no_aa (ablated denominator without A->A)."""
    heads = result["n_heads"]
    sizes = result["section_tokens"]
    n_answer = sizes["answer"]
    features = {}
    for layer, masses in result["layer_masses"].items():
        a = [m / (heads * n_answer) for m in masses[2]]  # answer prediction rows
        aAI, aAQ, aAV, aAR = a[0], a[1], a[2], a[3]
        aAA = a[4] if len(a) > 4 else 0.0

        # 1. Standard U_direct: denominator contains all past tokens (I + Q + V + R + A = 1.0)
        denom_full = aAI + aAQ + aAV + aAR + aAA
        u_direct = (aAV + aAR) / (denom_full + EPS)

        # 2. Ablated U_direct_no_aa: denominator excludes intra-answer self-attention A->A,
        # measuring the ratio of self-generated context (V+R) relative to all prior context (I+Q+V+R).
        denom_no_aa = aAI + aAQ + aAV + aAR
        u_direct_no_aa = (aAV + aAR) / (denom_no_aa + EPS)

        features[int(layer)] = {
            "U_direct": u_direct,
            "U_direct_no_aa": u_direct_no_aa,
        }
    return features
