"""ScienceQA benchmark loader."""

from __future__ import annotations

from datasets import load_dataset

from .base import Benchmark


class ScienceQA(Benchmark):
    benchmark_type = "multi_choice"

    def __init__(self, source: str = "derek-thomas/ScienceQA", split: str = "test"):
        self.ds = load_dataset(source)
        self.split = split

    def obtain_size(self) -> int:
        return len(self.ds[self.split])

    def retrieve(self, idx: int) -> dict | None:
        row = self.ds[self.split][idx]
        if row.get("image") is None:
            return None
        question = _format_choice_question(row["question"], row["choices"])
        return {
            "idx": idx,
            "img": row["image"],
            "question": question,
            "gt_ans": row["answer"],
            "choices": row["choices"],
            "num_c": len(row["choices"]),
        }


def _format_choice_question(question: str, choices: list[str]) -> str:
    choices_text = ""
    choice_numbers = []
    for idx, choice in enumerate(choices):
        choices_text += f"({idx}): {choice}\n"
        choice_numbers.append(str(idx))
    return (
        f"{question}\n{choices_text}\n"
        f"This is a single choice question, answer only with choice number in {', '.join(choice_numbers)}."
    )
