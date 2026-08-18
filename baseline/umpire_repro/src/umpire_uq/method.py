from __future__ import annotations

import math
import statistics

import torch
import torch.nn.functional as F


class UmpireMethod:
    required_responses = "samples"

    def __init__(self, *, jitter: float = 1e-8) -> None:
        if not math.isfinite(jitter) or jitter <= 0:
            raise ValueError("jitter must be finite and positive")
        self.jitter = jitter

    @property
    def runtime_config(self) -> dict:
        return {
            "name": "umpire",
            "paper": "arXiv:2602.24195v1",
            "official_repository": "daohieu17ctt/UMPIRE",
            "response_representation": "normalized_last_layer_final_response_token",
            "probability": "final_answer_joint_probability_from_generation_scores",
            "jitter": self.jitter,
            "alpha": "abs(median(logdet)/median(incoherence_sum))",
            "score": "official_logdet_plus_alpha_times_incoherence_sum",
        }

    def compute(self, *, question, greedy, samples) -> dict:
        embeddings = torch.tensor(
            [sample.final_hidden for sample in samples], dtype=torch.float64
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(samples):
            raise ValueError("response embeddings have an invalid shape")
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError("response embeddings are not finite")
        if torch.any(torch.linalg.vector_norm(embeddings, dim=1) == 0):
            raise ValueError("response embeddings must have non-zero norm")
        embeddings = F.normalize(embeddings, p=2, dim=1)
        gram = embeddings @ embeddings.T
        regularized = gram + self.jitter * torch.eye(
            len(samples), dtype=torch.float64
        )
        sign, logabsdet = torch.linalg.slogdet(regularized)
        if sign <= 0 or not torch.isfinite(logabsdet):
            raise FloatingPointError("regularized Gram matrix is not positive definite")
        logdet = float(logabsdet)
        probabilities = [
            math.exp(sample.sampling_sequence_log_prob) for sample in samples
        ]
        incoherence_sum = sum(1.0 - value for value in probabilities)
        return {
            "valid": True,
            "error": None,
            "score": None,
            "semantic_volume": logdet / (2 * len(samples)),
            "incoherence_mean": incoherence_sum / len(samples),
            "official_logdet": logdet,
            "official_incoherence_sum": incoherence_sum,
            "alpha": None,
        }

    @staticmethod
    def finalize(values: list[dict]) -> None:
        valid = [value for value in values if value.get("valid") is True]
        if not valid:
            return
        median_logdet = statistics.median(
            value["official_logdet"] for value in valid
        )
        median_incoherence = statistics.median(
            value["official_incoherence_sum"] for value in valid
        )
        if median_incoherence <= 0:
            raise ValueError("median UMPIRE incoherence must be positive")
        alpha = abs(median_logdet / median_incoherence)
        for value in valid:
            value["alpha"] = alpha
            value["score"] = value["official_logdet"] + alpha * value[
                "official_incoherence_sum"
            ]
