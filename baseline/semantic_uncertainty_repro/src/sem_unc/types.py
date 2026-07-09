"""Data types for semantic uncertainty pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SingleGeneration:
    """A single model generation."""

    answer: str
    token_log_likelihoods: list[float]
    embedding: "torch.Tensor | None" = None  # noqa: F821
    accuracy: float = 0.0


@dataclass
class SampleResult:
    """Per-sample result after generation + uncertainty computation."""

    id: str
    question: str
    gt_answers: list[str]
    most_likely_answer: SingleGeneration
    high_temp_answers: list[SingleGeneration] = field(default_factory=list)
    semantic_ids: list[int] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    correct: bool = False
    prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "gt_answers": self.gt_answers,
            "prediction": self.most_likely_answer.answer,
            "most_likely_answer": {
                "answer": self.most_likely_answer.answer,
                "token_log_likelihoods": self.most_likely_answer.token_log_likelihoods,
                "accuracy": self.most_likely_answer.accuracy,
            },
            "high_temp_answers": [
                {
                    "answer": gen.answer,
                    "token_log_likelihoods": gen.token_log_likelihoods,
                    "accuracy": gen.accuracy,
                }
                for gen in self.high_temp_answers
            ],
            "semantic_ids": self.semantic_ids,
            "correct": self.correct,
            "scores": self.scores,
            "prompt": self.prompt,
        }
