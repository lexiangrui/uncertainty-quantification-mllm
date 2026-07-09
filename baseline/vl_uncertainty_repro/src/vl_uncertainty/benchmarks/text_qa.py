"""Text QA datasets for semantic uncertainty verification."""

from __future__ import annotations

import hashlib

import datasets as hf_datasets

from .base import Benchmark


class TriviaQABenchmark(Benchmark):
    """TriviaQA in SQuAD format."""

    benchmark_type = "free_form"

    def __init__(
        self,
        source: str = "TimoImhof/TriviaQA-in-SQuAD-format",
        split: str = "validation",
        seed: int = 10,
    ):
        ds = hf_datasets.load_dataset(source)["unmodified"]
        split_ds = ds.train_test_split(test_size=0.2, seed=seed)
        self._data = split_ds["test" if split == "validation" else "train"]

    def obtain_size(self) -> int:
        return len(self._data)

    def retrieve(self, idx: int) -> dict | None:
        row = self._data[idx]
        answers = row["answers"]["text"]
        return {
            "idx": idx,
            "id": row.get("id", f"triviaqa-{idx}"),
            "img": None,
            "question": row["question"],
            "context": row.get("context", ""),
            "gt_ans": answers[0] if answers else "",
            "all_gt_ans": answers,
        }
