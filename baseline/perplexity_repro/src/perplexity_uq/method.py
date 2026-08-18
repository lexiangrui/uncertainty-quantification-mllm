from __future__ import annotations

import math


class PerplexityMethod:
    required_responses = "greedy"

    @property
    def runtime_config(self) -> dict:
        return {
            "name": "perplexity",
            "response": "greedy_final_answer",
            "probability": "transient_raw_generation_logits",
            "normalization": "token_mean_negative_log_likelihood",
        }

    def compute(self, *, question, greedy, samples) -> dict:
        nll = -greedy.mean_log_prob
        perplexity = math.exp(nll)
        if not math.isfinite(perplexity):
            raise FloatingPointError("perplexity is not finite")
        return {
            "valid": True,
            "error": None,
            "score": perplexity,
            "mean_nll": nll,
            "token_count": greedy.token_count,
        }
