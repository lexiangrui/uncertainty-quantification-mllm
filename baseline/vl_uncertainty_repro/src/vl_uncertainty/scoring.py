"""VL-Uncertainty scoring — faithful to Zhang et al. (2024).

Core algorithm
--------------
1. Generate 1 low-temperature answer as "most likely answer".
2. Apply 5 visual perturbations (Gaussian blur radii [0.6, 0.8, 1.0, 1.2, 1.4]).
3. Apply 5 textual perturbations (LLM rephrasing, temps [0.1, 0.2, 0.3, 0.4, 0.5]).
4. Pair visual and textual perturbations progressively (I_i ↔ T_i).
5. Generate 1 answer from each perturbed prompt at sampling temperature.
6. Cluster answers by semantic equivalence.
   - Multi-choice: cluster by extracted option number.
   - Free-form: cluster via LLM bidirectional entailment.
7. Compute count-based Shannon entropy: H = -Σ (count_c/N) log₂(count_c/N).

Reference: https://github.com/Ruiyang-061X/VL-Uncertainty
"""

from __future__ import annotations

import collections
import logging
import math
import re
from typing import Any

from vl_uncertainty.backends import Backend
from vl_uncertainty.perturbations import (
    PerturbationConfig,
    combine_perturbed_prompts,
    perturb_textual_prompt,
    perturb_visual_prompt,
)
from vl_uncertainty.text_models import TextModel
from vl_uncertainty.types import VLUncertaintyResult

logger = logging.getLogger(__name__)

# Default perturbation config matching the paper's optimal settings.
DEFAULT_PERT_CONFIG = PerturbationConfig(
    visual_perturbation="blurring",
    blur_radius_list=(0.6, 0.8, 1.0, 1.2, 1.4),
    textual_perturbation="llm_rephrasing",
    textual_temps=(0.1, 0.2, 0.3, 0.4, 0.5),
    sampling_time=5,
    pair_order="progressively",
)

# Entailment prompt from official VL-Uncertainty code.
ENTAILMENT_PROMPT = (
    "Does '{text1}' entail '{text2}'? "
    "Respond with either 'Yes' or 'No' only."
)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def compute_vl_uncertainty(
    backend: Backend,
    sample: dict[str, Any],
    text_model: TextModel,
    benchmark_type: str,
    pert_config: PerturbationConfig | None = None,
    sampling_temp: float = 1.0,
    max_new_tokens: int = 64,
) -> VLUncertaintyResult:
    """Compute VL-Uncertainty score for one sample.

    Parameters
    ----------
    backend : Backend
        LVLM backend (e.g. LLaVA).
    sample : dict
        Must contain ``"img"`` (PIL Image) and ``"question"`` (str).
    text_model : TextModel
        LLM used for both question rephrasing and entailment clustering.
    benchmark_type : str
        ``"multi_choice"`` or ``"free_form"``.
    pert_config : PerturbationConfig | None
        Perturbation parameters; defaults to the paper's optimal settings.
    sampling_temp : float
        Temperature for sampling from perturbed prompts.
    max_new_tokens : int
        Maximum new tokens per generation.

    Returns
    -------
    VLUncertaintyResult
    """
    cfg = pert_config or DEFAULT_PERT_CONFIG

    # ---- 1. Low-temperature most-likely answer ----
    ml_answer, _ml_log_liks, _ml_emb = backend.generate(
        sample["img"],
        sample["question"],
        temp=0.1,
        max_new_tokens=max_new_tokens,
    )

    # ---- 2. Visual perturbations ----
    pert_images = perturb_visual_prompt(sample["img"], cfg)

    # ---- 3. Textual perturbations ----
    pert_questions = perturb_textual_prompt(sample["question"], text_model, cfg)

    # ---- 4. Pair progressively (I_i ↔ T_i) ----
    pert_prompts = combine_perturbed_prompts(
        sample, pert_images, pert_questions, cfg.pair_order
    )

    # ---- 5. Generate 1 answer per perturbed prompt ----
    answers: list[str] = []
    for pp in pert_prompts:
        ans, _log_liks, _emb = backend.generate(
            pp["img"],
            pp["question"],
            temp=sampling_temp,
            max_new_tokens=max_new_tokens,
        )
        answers.append(ans)

    # ---- 6. Semantic clustering ----
    if benchmark_type == "multi_choice":
        num_c = int(sample.get("num_c", 0))
        cluster_ids = _cluster_multi_choice(answers, num_c)
    else:
        cluster_ids = _cluster_free_form(answers, text_model)

    # ---- 7. Count-based Shannon entropy ----
    uncertainty = _shannon_entropy(cluster_ids)

    distribution = {str(int(k)): int(v) for k, v in collections.Counter(cluster_ids).items()}

    return VLUncertaintyResult(
        uncertainty=float(uncertainty),
        cluster_ids=cluster_ids,
        cluster_distribution=distribution,
        sampled_answers=answers,
        perturbed_questions=pert_questions,
        most_likely_answer=ml_answer,
    )


# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------
def _cluster_free_form(answers: list[str], text_model: TextModel) -> list[int]:
    """Cluster free-form answers via LLM bidirectional entailment.

    Official algorithm from ``Ruiyang-061X/VL-Uncertainty``:
    for each unassigned answer i, assign it a new cluster ID, then for
    each subsequent answer j, check bidirectional entailment with i;
    if both directions are "Yes", assign j to the same cluster.
    """
    n = len(answers)
    semantic_ids = [-1] * n
    next_id = 0

    for i in range(n):
        if semantic_ids[i] != -1:
            continue
        semantic_ids[i] = next_id
        for j in range(i + 1, n):
            if semantic_ids[j] != -1:
                continue
            if _bidirectional_entailment(answers[i], answers[j], text_model):
                semantic_ids[j] = next_id
        next_id += 1

    assert -1 not in semantic_ids, f"Unassigned answers remain: {semantic_ids}"
    return semantic_ids


def _bidirectional_entailment(
    text1: str, text2: str, text_model: TextModel
) -> bool:
    """Check if text1 and text2 are semantically equivalent.

    Two directions must both return "Yes":
      - Does text1 entail text2?
      - Does text2 entail text1?
    """
    fwd = _llm_entailment_check(text1, text2, text_model)
    if not fwd:
        return False
    rev = _llm_entailment_check(text2, text1, text_model)
    return rev


def _llm_entailment_check(
    text1: str, text2: str, text_model: TextModel
) -> bool:
    """Single-direction entailment via LLM."""
    prompt = ENTAILMENT_PROMPT.format(text1=text1, text2=text2)
    response = text_model.generate(prompt, temp=0.1, max_new_tokens=8)
    return _is_yes(response)


def _is_yes(text: str) -> bool:
    """Check whether an LLM response means 'Yes'.

    Matches official code: ``'Yes'``, ``'yes'``, ``'Y'``, ``'y'``.
    """
    t = text.strip().lower()
    return t in ("yes", "y") or t.startswith("yes")


def _cluster_multi_choice(answers: list[str], num_choices: int) -> list[int]:
    """Cluster multi-choice answers by extracted option number."""
    ids: list[int] = []
    for ans in answers:
        m = re.search(r"\d+", ans or "")
        if m is None:
            ids.append(-1)
        else:
            c = int(m.group())
            ids.append(c if 0 <= c < num_choices else -1)
    return ids


# ------------------------------------------------------------------
# Entropy
# ------------------------------------------------------------------
def _shannon_entropy(cluster_ids: list[int]) -> float:
    """Count-based Shannon entropy: H = -Σ p(c) log₂ p(c).

    Uses log base 2, matching the official VL-Uncertainty implementation.
    """
    n = len(cluster_ids)
    if n == 0:
        return 0.0
    counts = collections.Counter(cluster_ids)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / n
        if p > 0:
            entropy -= p * math.log2(p)
    return max(entropy, 0.0)
