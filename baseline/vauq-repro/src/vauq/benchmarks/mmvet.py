"""MMVet benchmark (free-form)."""

from __future__ import annotations

from datasets import load_dataset

from .base import Benchmark


class MMVet(Benchmark):
    def __init__(
        self,
        source: str = "whyu/mm-vet",
        split: str = "test",
        question_style: str = "final_answer",
    ):
        self.ds = load_dataset(source)
        self.split = split
        STYLES = {
            "raw": lambda q: q,
            "final_answer": lambda q: (
                f"{q}\n"
                "NOTE: Provide only the final answer. Do not provide unrelated details."
            ),
            "describe_then_answer": lambda q: (
                f"{q}\n"
                "First note what you see in the image relevant to the question, then answer."
            ),
        }
        if question_style not in STYLES:
            raise ValueError(f"unknown MMVet question_style: {question_style!r}")
        self._format_question = STYLES[question_style]
        self.question_style = question_style

    def obtain_size(self) -> int:
        return len(self.ds[self.split])

    def retrieve(self, idx: int) -> dict | None:
        row = self.ds[self.split][idx]
        raw_question = str(row["question"])
        question = self._format_question(raw_question)
        return {
            "idx": idx,
            "img": row["image"],
            "question": question,
            "raw_question": raw_question,
            "gt_ans": row["answer"],
        }
