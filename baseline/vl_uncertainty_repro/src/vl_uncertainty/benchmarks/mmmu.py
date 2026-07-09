"""MMMU benchmark loader."""

from __future__ import annotations

import ast

from datasets import load_dataset

from .base import Benchmark


class MMMU(Benchmark):
    benchmark_type = "multi_choice"

    categories = [
        "Accounting",
        "Agriculture",
        "Architecture_and_Engineering",
        "Art",
        "Art_Theory",
        "Basic_Medical_Science",
        "Biology",
        "Chemistry",
        "Clinical_Medicine",
        "Computer_Science",
        "Design",
        "Diagnostics_and_Laboratory_Medicine",
        "Economics",
        "Electronics",
        "Energy_and_Power",
        "Finance",
        "Geography",
        "History",
        "Literature",
        "Manage",
        "Marketing",
        "Materials",
        "Math",
        "Mechanical_Engineering",
        "Music",
        "Pharmacy",
        "Physics",
        "Psychology",
        "Public_Health",
        "Sociology",
    ]

    def __init__(self, source: str = "MMMU/MMMU", split: str = "validation", samples_each_category: int = 30):
        self.source = source
        self.split = split
        self.samples_each_category = samples_each_category
        self.ds_map = {category: load_dataset(source, category) for category in self.categories}

    def obtain_size(self) -> int:
        return len(self.categories) * self.samples_each_category

    def retrieve(self, idx: int) -> dict | None:
        ds_id = int(idx / self.samples_each_category)
        real_index = idx % self.samples_each_category
        category = self.categories[ds_id]
        row = self.ds_map[category][self.split][real_index]
        if "<image 2>" in row["question"] or row.get("question_type") == "open":
            return None
        options = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]
        question = _format_choice_question(row["question"], options)
        mapping = {letter: number for number, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
        return {
            "idx": idx,
            "img": row["image_1"],
            "question": question,
            "gt_ans": mapping[row["answer"]],
            "choices": options,
            "num_c": len(options),
            "subset": category,
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
