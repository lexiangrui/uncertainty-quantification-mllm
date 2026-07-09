"""MM-Vet benchmark loader."""

from __future__ import annotations

from datasets import load_dataset

from .base import Benchmark


class MMVet(Benchmark):
    benchmark_type = "free_form"

    def __init__(self, source: str = "whyu/mm-vet", split: str = "test"):
        self.ds = load_dataset(source)
        self.split = split

    def obtain_size(self) -> int:
        return len(self.ds[self.split])

    def retrieve(self, idx: int) -> dict | None:
        row = self.ds[self.split][idx]
        question = (
            f"{row['question']}\n"
            "NOTE: Provide only the final answer. Do not provide unrelated details."
        )
        return {"idx": idx, "img": row["image"], "question": question, "gt_ans": row["answer"]}
