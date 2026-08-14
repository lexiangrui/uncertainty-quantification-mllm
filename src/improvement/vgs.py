"""Visual Grounding Score (VGS) — attention-based UQ.

Measures the fraction of attention from answer tokens directed at visual
tokens vs. text tokens.  Low visual grounding → answer generated from
language context → hallucination risk.

Uses a **streaming hook approach**: instead of ``output_attentions=True``
(which stores ALL layers' attention simultaneously and OOMs on long
sequences), a forward hook on each selected attention layer processes
and discards the attention weights immediately.  Peak memory is one
layer's attention matrix, not all layers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from src.improvement.lac import LacBackend


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
            "vision_attn_mean": self.vision_attn_mean,
            "n_answer_tokens": self.n_answer_tokens,
            "n_visual_tokens": self.n_visual_tokens,
        }


def _get_decoder_layers(backend: LacBackend):
    """Return the list of decoder layers from the model."""
    base = backend._base_model()
    core = getattr(base, "model", base)
    lang_model = getattr(core, "language_model", core)
    decoder = getattr(lang_model, "model", lang_model)
    layers = getattr(decoder, "layers", None)
    if layers is not None:
        return list(layers)
    raise RuntimeError("cannot locate decoder layers")


def compute_vgs(
    backend: LacBackend,
    full_inputs: dict,
    prompt_length: int,
    answer_span: tuple[int, int],
) -> VgsResult:
    """Compute Visual Grounding Score using streaming attention hooks."""
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

    predict_positions = list(range(ans_start - 1, ans_end - 1))
    visual_positions = visual_mask.nonzero(as_tuple=True)[0]

    layers = _get_decoder_layers(backend)
    n_layers = len(layers)
    layer_start = n_layers // 3
    layer_end = n_layers

    # Streaming accumulation: process each layer's attention in a hook,
    # then discard it immediately to avoid OOM on long sequences.
    total_visual_attn = 0.0
    total_all_attn = 0.0

    # Pre-compute tensors on the correct device for use inside hooks
    predict_idx = torch.tensor(predict_positions, dtype=torch.long)
    visual_idx = visual_positions.cpu()

    def make_hook():
        def hook(module, args, output):
            nonlocal total_visual_attn, total_all_attn
            # output from eager attention: (attn_output, attn_weights) or more
            if not isinstance(output, tuple) or len(output) < 2:
                return
            attn_weights = output[1]
            if attn_weights is None:
                return
            # attn_weights: (batch, heads, seq, seq)
            # Extract attention from predict_positions to all positions
            attn = attn_weights[0]  # (heads, seq, seq)
            dev = attn.device
            pidx = predict_idx.to(dev)
            vidx = visual_idx.to(dev)

            ans_attn = attn[:, pidx, :]  # (heads, n_ans, seq)
            ans_sum = ans_attn.sum(dim=(0, 1))  # (seq,)
            total_visual_attn += ans_sum[vidx].sum().item()
            total_all_attn += ans_sum.sum().item()

            # Discard attention to free memory — return None in its place
            output_list = list(output)
            output_list[1] = None
            return tuple(output_list)
        return hook

    # Register hooks on selected layers
    handles = []
    for layer_idx in range(layer_start, min(layer_end, n_layers)):
        layer = layers[layer_idx]
        attn_module = getattr(layer, "self_attn", None)
        if attn_module is not None:
            h = attn_module.register_forward_hook(make_hook())
            handles.append(h)

    try:
        with torch.inference_mode():
            outputs = backend.model(
                **full_inputs,
                output_attentions=True,  # needed so attention is computed
                use_cache=False,
            )
    finally:
        for h in handles:
            h.remove()

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vision_ratio = total_visual_attn / max(total_all_attn, 1e-12)
    score = -vision_ratio

    return VgsResult(
        score=score,
        vision_attn_ratio=vision_ratio,
        vision_attn_mean=vision_ratio,
        n_answer_tokens=n_ans,
        n_visual_tokens=n_visual,
    )
