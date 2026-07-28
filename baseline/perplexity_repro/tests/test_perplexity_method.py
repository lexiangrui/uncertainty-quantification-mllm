from __future__ import annotations

import math
from types import SimpleNamespace

from perplexity_uq import PerplexityMethod


def test_perplexity_uses_greedy_answer_mean_nll() -> None:
    greedy = SimpleNamespace(mean_log_prob=-math.log(2), token_count=3)
    result = PerplexityMethod().compute(
        question="question", greedy=greedy, samples=[]
    )
    assert math.isclose(result["score"], 2.0)
    assert math.isclose(result["mean_nll"], math.log(2))
    assert result["token_count"] == 3
