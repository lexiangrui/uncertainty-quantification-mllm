from __future__ import annotations

from datasets import concatenate_datasets, load_dataset

from .base import Benchmark


class CVBench(Benchmark):
    benchmark_type = "multi_choice"

    def __init__(self, source: str = "nyu-visionx/CV-Bench", split: str = "test"):
        self.ds = concatenate_datasets([load_dataset(source, config)[split] for config in ("2D", "3D")])

    def obtain_size(self) -> int:
        return len(self.ds)

    def retrieve(self, idx: int) -> dict:
        row = self.ds[idx]
        choices = [str(choice) for choice in row["choices"]]
        options = "\n".join(f"({i}): {choice}" for i, choice in enumerate(choices))
        gold_letter = str(row["answer"]).strip().strip("()")
        gold_index = ord(gold_letter.upper()) - ord("A")
        return {
            "idx": idx,
            "img": row["image"],
            "question": f"{row['question']}\n{options}\nAnswer only with the choice number.",
            "gt_ans": gold_index,
            "choices": choices,
            "subset": row.get("type"),
        }
