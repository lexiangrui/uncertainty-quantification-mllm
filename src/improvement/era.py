"""ERA (Early Rationale Attribution) — single-pass attention-based uncertainty quantification.

Quantifies the degree to which an MLLM's final answer relies on its own
self-generated rationale (<vision> and <reasoning>) versus ground-truth external
inputs (image visual tokens and user prompt text) in early decoder layers (0-1).

Five destination regions are distinguished:
    I  image visual tokens         (ground-truth visual evidence)
    Q  prompt-text tokens          (system prompt & question text)
    V  <vision>…</vision>          (self-generated visual description slice)
    R  <reasoning>…</reasoning>    (self-generated reasoning chain slice)
    A  <answer>…</answer>          (self-generated answer slice)

For every decoder layer and every source group S ∈ {V, R, A}, we accumulate the
attention mass from prediction rows (row t-1 predicts token t) to destination regions:
    M[l][S][T] = Σ_{rows t∈S} Σ_{heads h} Σ_{j∈T} A^(l,h)[t-1, j]

Canonical ERA metric U_ERA:
    U_ERA = (α(A→V) + α(A→R)) / (α(A→I) + α(A→Q) + α(A→V) + α(A→R) + ε)

Excludes intra-answer causal self-attention (A→A) from the denominator,
guaranteeing response length invariance and measuring the pure relative odds of
internal ungrounded dependency vs external ground-truth anchoring.
"""
from __future__ import annotations

import torch
from dataclasses import dataclass

from src.improvement.backend import EraBackend

_CHUNK = 256

GROUPS = ("vision", "reasoning", "answer")
DESTS = ("image", "prompt_text", "vision", "reasoning", "answer")
FEATURES = ("U_ERA", "U_direct")
EPS = 1e-8


@dataclass
class EraResult:
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


def _get_decoder_layers(backend: EraBackend):
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


class _EraAccumulator:
    """Accumulates group→bucket attention mass per decoder layer."""

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
        if probs.shape[3] != self.col_bucket.numel():
            raise RuntimeError(
                "ERA attention/layout mismatch: "
                f"KV length={probs.shape[3]}, col_bucket length={self.col_bucket.numel()}"
            )
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


def _make_chunked_eager(accumulator: _EraAccumulator):
    def chunked_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        is_causal = kwargs.get("is_causal")
        is_decoder = id(module) in accumulator.module_layers
        groups = getattr(module, "num_key_value_groups", 1)
        key_states = _repeat_kv(key, groups)
        value_states = _repeat_kv(value, groups)
        q_len = query.shape[2]
        kv_len = key_states.shape[2]

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
                rows = torch.arange(s, e, device=scores.device).unsqueeze(1)
                cols = torch.arange(kv_len, device=scores.device).unsqueeze(0)
                causal = cols <= rows
                scores = scores.masked_fill(
                    ~causal.view(1, 1, e - s, kv_len), torch.finfo(scores.dtype).min
                )
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
            out[:, :, s:e, :] = torch.matmul(probs.to(value_states.dtype), value_states)
            accumulator.accumulate(module, probs, s)
        return out.transpose(1, 2).contiguous(), None

    return chunked_eager


def compute_era(
    backend: EraBackend,
    full_inputs: dict,
    prompt_length: int,
    generated_buckets: list[int],
) -> EraResult | None:
    """Compute per-layer direct-attention masses for ERA."""
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

    visual_mask = input_ids[0] == image_token_id
    n_visual = int(visual_mask.sum().item())
    if n_visual == 0:
        return None

    # Prediction rows per group: row t-1 predicts token t.
    predict_idx = torch.cat([positions[name] - 1 for name in GROUPS])
    row_groups = torch.cat([
        torch.full((positions[name].numel(),), group, dtype=torch.long)
        for group, name in enumerate(GROUPS)
    ])

    # The processor must expose the exact decoder attention layout in input_ids.
    col_bucket = torch.full((seq_len,), 1, dtype=torch.long)
    col_bucket[visual_mask.cpu()] = 0
    col_bucket[prompt_length:] = generated_bucket_tensor

    layers = _get_decoder_layers(backend)
    accumulator = _EraAccumulator(predict_idx, row_groups, col_bucket)
    accumulator.map_modules(layers)

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

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
    return EraResult(
        n_visual_tokens=n_visual,
        n_heads=accumulator.n_heads,
        section_tokens={name: positions[name].numel() for name in GROUPS},
        layer_masses=accumulator.layer_masses,
    )


# Default early decoder layers evaluated in the method
DIRECT_LAYERS = (0, 1)


def layer_features(result: dict) -> dict[int, dict[str, float]]:
    """Compute per-layer ERA uncertainty features (without intra-answer A->A)."""
    heads = result["n_heads"]
    sizes = result["section_tokens"]
    n_answer = sizes["answer"]
    features = {}
    for layer, masses in result["layer_masses"].items():
        a = [m / (heads * n_answer) for m in masses[2]]  # answer prediction rows
        aAI, aAQ, aAV, aAR = a[0], a[1], a[2], a[3]

        # Canonical ERA score:
        # Ratio of self-generated rationale attention (V+R) to total prior context (I+Q + V+R).
        denom = aAI + aAQ + aAV + aAR
        score = (aAV + aAR) / (denom + EPS)

        features[int(layer)] = {
            "U_ERA": score,
            "U_direct": score,  # alias for backward compatibility
        }
    return features
