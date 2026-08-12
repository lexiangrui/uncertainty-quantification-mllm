"""Visual Grounding Score (VGS) — attention-based UQ.

Measures the fraction of attention from answer tokens directed at visual
tokens vs. text tokens.  Low visual grounding → answer generated from
language context → hallucination risk.

**IMPORTANT**: Requires ``attn_implementation='eager'`` because SDPA does
not return attention weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from src.improvement.lac import LacBackend, LacResult


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


def compute_vgs(
    backend: LacBackend,
    full_inputs: dict,
    prompt_length: int,
    answer_span: tuple[int, int],
) -> VgsResult:
    """Compute Visual Grounding Score from attention weights."""
    input_ids = full_inputs["input_ids"]
    image_token_id = backend._image_token_id
    ans_start, ans_end = answer_span
    n_ans = ans_end - ans_start
    if n_ans < 1:
        return VgsResult(None, None, None, 0, 0)

    # Identify visual token positions in the full sequence
    visual_mask = (input_ids[0] == image_token_id)
    n_visual = visual_mask.sum().item()
    if n_visual == 0:
        return VgsResult(None, None, None, n_ans, 0)

    # Forward pass with attentions
    with torch.inference_mode():
        outputs = backend.model(**full_inputs, output_attentions=True, use_cache=False)

    attentions = outputs.attentions  # tuple of (1, heads, seq, seq) per layer
    if attentions is None:
        del outputs
        return VgsResult(None, None, None, n_ans, n_visual)

    n_layers = len(attentions)
    # Use middle-to-late layers (VAUQ found these most informative)
    layer_start = n_layers // 3
    layer_end = n_layers

    predict_positions = list(range(ans_start - 1, ans_end - 1))  # positions that predict answer tokens
    visual_positions = visual_mask.nonzero(as_tuple=True)[0]

    # Aggregate attention from answer-predicting positions to visual positions
    total_visual_attn = 0.0
    total_all_attn = 0.0

    for layer_idx in range(layer_start, layer_end):
        attn = attentions[layer_idx][0]  # (heads, seq, seq)
        # Attention from predict_positions to all positions
        ans_attn = attn[:, predict_positions, :]  # (heads, n_ans, seq)
        # Sum over heads and answer positions
        ans_attn_sum = ans_attn.sum(dim=(0, 1))  # (seq,)
        total_visual_attn += ans_attn_sum[visual_positions].sum().item()
        total_all_attn += ans_attn_sum.sum().item()

    del outputs, attentions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vision_ratio = total_visual_attn / max(total_all_attn, 1e-12)
    # Score: LOW vision_ratio = high uncertainty (answer ignores visual input)
    # Invert so higher score = more uncertain
    score = -vision_ratio  # negative: more negative = more visual grounding = less uncertain

    return VgsResult(
        score=score,
        vision_attn_ratio=vision_ratio,
        vision_attn_mean=vision_ratio,  # same as ratio for now
        n_answer_tokens=n_ans,
        n_visual_tokens=n_visual,
    )
