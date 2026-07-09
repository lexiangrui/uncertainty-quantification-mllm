"""CV-Bench benchmark (multiple-choice, 2D + 3D subsets)."""

from __future__ import annotations

from datasets import concatenate_datasets, load_dataset

from .base import Benchmark


class CVBench(Benchmark):
    """CV-Bench, default both 2D and 3D subsets concatenated (2638 examples).

    Each row carries a ``type`` field of ``"2D"`` / ``"3D"`` used as the subset
    tag (exposed via ``sample["subset"]``).
    """

    def __init__(
        self,
        source: str = "nyu-visionx/CV-Bench",
        configs: tuple[str, ...] = ("2D", "3D"),
        split: str = "test",
    ):
        parts = [load_dataset(source, c)[split] for c in configs]
        self.ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        self.configs = list(configs)

    def obtain_size(self) -> int:
        return len(self.ds)

    def retrieve(self, idx: int) -> dict | None:
        row = self.ds[idx]
        question = (
            f"{row['prompt']}\n"
            "Your answer should be only the letter of the option."
        )
        return {
            "idx": idx,
            "img": row["image"],
            "question": question,
            "gt_ans": row["answer"],
            "choices": row["choices"],
            "subset": row.get("type"),
        }
