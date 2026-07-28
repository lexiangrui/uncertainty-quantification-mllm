import re
from io import BytesIO
from pathlib import Path
from typing import Iterator

import pandas as pd
from PIL import Image

from config import DATA_ROOT


CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FREE_FORM_INSTRUCTION = "NOTE: Provide only the final answer. Do not provide unrelated details."


def format_multiple_choice(question: str, choices: list[str]) -> str:
    if not 2 <= len(choices) <= len(CHOICE_LETTERS):
        raise ValueError(f"invalid number of choices: {len(choices)}")
    options = "\n".join(f"({CHOICE_LETTERS[index]}) {choice}" for index, choice in enumerate(choices))
    return f"{question.strip()}\n{options}\nAnswer with only the option letter."


def format_question(question: str, choices: list[str] | None) -> str:
    if choices:
        return format_multiple_choice(question, choices)
    return f"{question.strip()}\n{FREE_FORM_INSTRUCTION}"


def _cvbench_files(root: Path, subset: str | None = None) -> list[Path]:
    base = root / "nyu-visionx___cv-bench"
    pattern = f"{subset}/0.0.0/*/cv-bench-test.arrow" if subset else "*/0.0.0/*/cv-bench-test.arrow"
    files = sorted(base.glob(pattern))
    expected = 1 if subset else 2
    if len(files) != expected:
        raise FileNotFoundError(f"expected {expected} CVBench arrow file(s), found {files}")
    return files


def iter_cvbench(root: Path, limit: int | None = None, subset: str | None = None) -> Iterator[dict]:
    from datasets import Dataset

    emitted = 0
    for arrow_file in _cvbench_files(root, subset):
        dataset = Dataset.from_file(str(arrow_file))
        for row in dataset:
            choices = [str(item) for item in row["choices"]]
            answer_match = re.fullmatch(r"\(([A-Z])\)", row["answer"].strip().upper())
            if answer_match is None:
                raise ValueError(f"invalid CVBench answer: {row['answer']!r}")
            letter = answer_match.group(1)
            answer_index = CHOICE_LETTERS.index(letter)
            if answer_index >= len(choices):
                raise ValueError(f"answer outside choices: {row['answer']!r}")
            yield {
                "id": f"cvbench-{row['type'].lower()}-{row['idx']}",
                "dataset": "cvbench",
                "image": row["image"].convert("RGB"),
                "question_stem": row["question"].strip(),
                "question": format_multiple_choice(row["question"], choices),
                "references": [letter, choices[answer_index]],
                "choices": choices,
                "answer_index": answer_index,
                "metadata": {"type": row["type"], "task": row["task"], "source": row["source"]},
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_mmvet(root: Path, limit: int | None = None) -> Iterator[dict]:
    from datasets import Dataset

    files = sorted((root / "whyu___mm-vet").glob("**/mm-vet-test.arrow"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one MM-Vet arrow file, found {files}")
    dataset = Dataset.from_file(str(files[0]))
    for index, row in enumerate(dataset):
        question = str(row["question"]).strip()
        yield {
            "id": f"mmvet-{index}",
            "dataset": "mmvet",
            "image": row["image"].convert("RGB"),
            "question_stem": question,
            "question": format_question(question, None),
            "references": [str(row["answer"])],
            "choices": None,
            "answer_index": None,
            "metadata": {"source_id": row["id"], "capability": row.get("capability")},
        }
        if limit is not None and index + 1 >= limit:
            return


def iter_vilp(root: Path, limit: int | None = None) -> Iterator[dict]:
    path = root / "vilp" / "ViLP.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    dataframe = pd.read_parquet(path)
    emitted = 0
    for original_index, row in dataframe.iterrows():
        question = str(row["question"]).strip()
        for case in (1, 2):
            image = Image.open(BytesIO(row[f"image{case}"])).convert("RGB")
            yield {
                "id": f"vilp-{original_index}-case{case}",
                "dataset": "vilp",
                "image": image,
                "question_stem": question,
                "question": format_question(question, None),
                "references": [str(row[f"answer{case}"])],
                "choices": None,
                "answer_index": None,
                "metadata": {"original_index": int(original_index), "case": case},
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_samples(dataset_name: str, split: str = "test", limit: int | None = None) -> Iterator[dict]:
    if split != "test":
        raise ValueError("the current datasets expose only the test split")
    if dataset_name == "cvbench":
        yield from iter_cvbench(DATA_ROOT, limit)
        return
    if dataset_name == "cvbench2d":
        yield from iter_cvbench(DATA_ROOT, limit, subset="2D")
        return
    if dataset_name == "mmvet":
        yield from iter_mmvet(DATA_ROOT, limit)
        return
    if dataset_name == "vilp":
        yield from iter_vilp(DATA_ROOT, limit)
        return
    raise ValueError(f"dataset is not implemented yet: {dataset_name}")
