"""Semantic entropy computation.

Core algorithm from Farquhar et al. (Nature 2024):

1. Sample multiple high-temperature generations.
2. Cluster them by semantic equivalence (bi-directional entailment).
3. Compute entropy over the meaning-clusters.

Also provides naive (regular) entropy and cluster-assignment entropy
for comparison.
"""

from __future__ import annotations

import logging

import numpy as np

from sem_unc.entailment import EntailmentModel

logger = logging.getLogger(__name__)


def get_semantic_ids(
    responses: list[str],
    entailment_model: EntailmentModel,
    strict_entailment: bool = False,
) -> list[int]:
    """Group a list of response strings into semantic clusters.

    Two responses are semantically equivalent if each entails the other.

    Parameters
    ----------
    responses:
        List of generated answer strings.
    entailment_model:
        Model with a ``check_implication(text1, text2) -> int`` method.
    strict_entailment:
        If ``True``, both directions must be ``2`` (entailment).
        If ``False`` (default), accept as long as neither direction is
        ``0`` (contradiction) and not both ``1`` (neutral).

    Returns
    -------
    semantic_ids : list[int]
        Cluster assignment for each response.  IDs are 0-indexed and
        contiguous.
    """

    def _are_equivalent(text1: str, text2: str) -> bool:
        imp_1 = entailment_model.check_implication(text1, text2)
        imp_2 = entailment_model.check_implication(text2, text1)

        if strict_entailment:
            return imp_1 == 2 and imp_2 == 2

        # Non-strict: neither contradicts, and not both neutral.
        return (0 not in (imp_1, imp_2)) and ((imp_1, imp_2) != (1, 1))

    n = len(responses)
    semantic_ids = [-1] * n
    next_id = 0

    for i in range(n):
        if semantic_ids[i] != -1:
            continue
        semantic_ids[i] = next_id
        for j in range(i + 1, n):
            if semantic_ids[j] == -1 and _are_equivalent(responses[i], responses[j]):
                semantic_ids[j] = next_id
        next_id += 1

    assert -1 not in semantic_ids, "Some responses were not assigned a cluster."
    return semantic_ids


def logsumexp_by_id(
    semantic_ids: list[int],
    log_likelihoods: list[float],
) -> list[float]:
    """Aggregate token log-likelihoods by semantic cluster.

    For each cluster, compute the log-sum-exp of the length-normalised
    log-likelihoods of all responses assigned to that cluster, then
    re-normalise so the per-cluster values sum to 1 in probability space.

    Returns log-probabilities, one per unique cluster.
    """
    unique_ids = sorted(set(semantic_ids))
    n = len(log_likelihoods)

    # Normalize: subtract log-sum-exp of all responses.
    log_total = np.log(np.sum(np.exp(log_likelihoods)))

    log_prob_per_id = []
    for uid in unique_ids:
        indices = [pos for pos, sid in enumerate(semantic_ids) if sid == uid]
        id_log_liks = [log_likelihoods[i] for i in indices]
        # log( sum( exp(log_lik_norm) ) ) where log_lik_norm = l - log_total
        id_log_liks_arr = np.array(id_log_liks)
        logsumexp = np.log(np.sum(np.exp(id_log_liks_arr - log_total)))
        log_prob_per_id.append(logsumexp)

    return log_prob_per_id


def predictive_entropy(log_probs: list[float]) -> float:
    """Naive (regular) predictive entropy.

    E[-log p] ≈ -1/N Σ log p(x_i) = average negative log-likelihood.
    """
    if len(log_probs) == 0:
        return 0.0
    return float(-np.sum(log_probs) / len(log_probs))


def predictive_entropy_rao(log_probs: list[float]) -> float:
    """Rao-Blackwellised semantic entropy.

    H = -Σ p_c * log p_c  where p_c is the probability mass of cluster c
    (in log space as input).
    """
    probs = np.exp(log_probs)
    # Guard against numerical issues.
    probs = probs / probs.sum()
    entropy = -np.sum(probs * log_probs)
    return float(max(entropy, 0.0))


def cluster_assignment_entropy(semantic_ids: list[int]) -> float:
    """Entropy of the categorical distribution over cluster assignments.

    Does *not* use token likelihoods — purely based on how often each
    cluster is sampled.
    """
    n = len(semantic_ids)
    if n == 0:
        return 0.0
    counts = np.bincount(semantic_ids)
    probs = counts / n
    # Ignore zero-probability clusters.
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def compute_semantic_entropy(
    responses: list[str],
    log_likelihoods: list[list[float]],
    entailment_model: EntailmentModel,
    strict_entailment: bool = False,
    question: str = "",
) -> dict[str, float]:
    """Compute all uncertainty measures for a set of sampled responses.

    Parameters
    ----------
    responses:
        High-temperature answer strings.
    log_likelihoods:
        Per-response list of per-token log-probabilities.
    entailment_model:
        Model for checking semantic equivalence.
    strict_entailment:
        Passed to ``get_semantic_ids``.

    Returns
    -------
    scores : dict
        Keys: ``"semantic_entropy"``, ``"regular_entropy"``,
        ``"cluster_assignment_entropy"``.
    """
    if len(responses) == 0:
        return {
            "semantic_entropy": 0.0,
            "regular_entropy": 0.0,
            "cluster_assignment_entropy": 0.0,
        }

    # 1. Semantic clustering.
    clustering_responses = [f"{question} {response}" for response in responses]
    semantic_ids = get_semantic_ids(
        clustering_responses, entailment_model, strict_entailment=strict_entailment
    )

    # 2. Length-normalised log-likelihoods (average per token).
    log_liks_agg = [np.mean(ll) for ll in log_likelihoods]

    # 3. Semantic entropy.
    log_prob_per_cluster = logsumexp_by_id(semantic_ids, log_liks_agg)
    sem_ent = predictive_entropy_rao(log_prob_per_cluster)

    # 4. Regular (naive) entropy.
    reg_ent = predictive_entropy(log_liks_agg)

    # 5. Cluster assignment entropy (no likelihoods).
    clust_ent = cluster_assignment_entropy(semantic_ids)

    return {
        "semantic_entropy": sem_ent,
        "regular_entropy": reg_ent,
        "cluster_assignment_entropy": clust_ent,
        "semantic_ids": semantic_ids,
    }
