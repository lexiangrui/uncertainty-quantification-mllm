from __future__ import annotations

import math
from dataclasses import dataclass

from src.llm_judge.nli import PairwiseNLIJudge


@dataclass(frozen=True)
class SemanticEntropyResult:
    semantic_entropy: float
    semantic_ids: tuple[int, ...]
    cluster_log_probs: tuple[float, ...]
    cluster_probs: tuple[float, ...]


class SemanticEntropyMethod:
    required_responses = "samples"

    def __init__(self, entailment: PairwiseNLIJudge) -> None:
        self.entailment = entailment

    @property
    def runtime_config(self) -> dict:
        return {
            "name": "semantic_entropy",
            "entailment_model": self.entailment.model_id,
            "equivalence": "exact_text_or_strict_bidirectional_entailment",
            "probability": "answer_mean_log_prob_from_transient_raw_logits",
        }

    def compute(
        self,
        *,
        question: str,
        greedy,
        samples,
    ) -> dict:
        answers = [sample.answer for sample in samples]
        mean_log_probs = [sample.mean_log_prob for sample in samples]
        result = compute_semantic_entropy(
            question, answers, mean_log_probs, self.entailment
        )
        clusters = []
        for cluster_id, probability in enumerate(result.cluster_probs):
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "members": [
                        index
                        for index, semantic_id in enumerate(result.semantic_ids)
                        if semantic_id == cluster_id
                    ],
                    "probability": probability,
                }
            )
        return {
            "valid": True,
            "error": None,
            "score": result.semantic_entropy,
            "semantic_ids": list(result.semantic_ids),
            "clusters": clusters,
        }


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def cluster_answers(
    question: str,
    answers: list[str],
    entailment: PairwiseNLIJudge,
) -> list[int]:
    if not answers or any(not answer.strip() for answer in answers):
        raise ValueError("answers must be non-empty")
    contextualized = [f"Question: {question}\nAnswer: {answer}" for answer in answers]
    semantic_ids = [-1] * len(answers)
    representatives: list[int] = []
    for index, answer in enumerate(contextualized):
        if not representatives:
            semantic_ids[index] = 0
            representatives.append(index)
            continue
        exact_matches = {
            cluster_id
            for cluster_id, representative in enumerate(representatives)
            if contextualized[representative] == answer
        }
        pairs = []
        for cluster_id, representative in enumerate(representatives):
            if cluster_id in exact_matches:
                continue
            pairs.append((answer, contextualized[representative]))
            pairs.append((contextualized[representative], answer))
        decisions = entailment.check_pairs(pairs) if pairs else []
        if len(decisions) != len(pairs):
            raise RuntimeError("entailment model returned an invalid number of decisions")
        assigned = None
        checked = 0
        for cluster_id in range(len(representatives)):
            # identical text is semantically equivalent by definition; the
            # entailment model is only consulted for non-identical pairs
            if cluster_id in exact_matches:
                assigned = cluster_id
                break
            if decisions[2 * checked] and decisions[2 * checked + 1]:
                assigned = cluster_id
                break
            checked += 1
        if assigned is None:
            assigned = len(representatives)
            representatives.append(index)
        semantic_ids[index] = assigned
    return semantic_ids


def compute_semantic_entropy(
    question: str,
    answers: list[str],
    mean_log_probs: list[float],
    entailment: PairwiseNLIJudge,
) -> SemanticEntropyResult:
    if len(answers) != len(mean_log_probs) or not answers:
        raise ValueError("answers and mean_log_probs must have the same non-zero length")
    if any(not math.isfinite(value) for value in mean_log_probs):
        raise ValueError("mean_log_probs must be finite")
    semantic_ids = cluster_answers(question, answers, entailment)
    normalization = _logsumexp(mean_log_probs)
    cluster_log_probs = []
    for cluster_id in range(max(semantic_ids) + 1):
        members = [
            mean_log_probs[index]
            for index, semantic_id in enumerate(semantic_ids)
            if semantic_id == cluster_id
        ]
        cluster_log_probs.append(_logsumexp(members) - normalization)
    cluster_probs = [math.exp(value) for value in cluster_log_probs]
    semantic_entropy = -sum(
        probability * log_probability
        for probability, log_probability in zip(cluster_probs, cluster_log_probs)
    )
    return SemanticEntropyResult(
        semantic_entropy=semantic_entropy,
        semantic_ids=tuple(semantic_ids),
        cluster_log_probs=tuple(cluster_log_probs),
        cluster_probs=tuple(cluster_probs),
    )
