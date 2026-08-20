from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ._image import decode_image, require_columns, required_text
from .base import BenchmarkSample


FILES = {
    "image": "image-00000-of-00001.parquet",
    "non_image": "non_image-00000-of-00001.parquet",
}
IDENTITY_COLUMNS = ("category", "subcategory", "set_id", "figure_id", "question_id")


def _yes_no(value: object, sample_id: str) -> str:
    normalized = str(value).strip().lower()
    mapping = {"0": "No", "1": "Yes", "no": "No", "yes": "Yes"}
    if normalized not in mapping:
        raise ValueError(f"invalid HallusionBench gt_answer for {sample_id}: {value!r}")
    return mapping[normalized]


def iter_hallusionbench(directory: Path) -> Iterator[BenchmarkSample]:
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    for split, filename in FILES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        required = {
            "question",
            "gt_answer",
            "gt_answer_details",
            "visual_input",
        } | set(IDENTITY_COLUMNS)
        require_columns(list(frame.columns), required, str(path))
        for row_number, row in frame.iterrows():
            identity = "-".join(str(row[name]).strip() for name in IDENTITY_COLUMNS)
            sample_id = f"hallusionbench-{split}-{identity}"
            value = row["visual_input"]
            if isinstance(value, bool):
                visual_input = value
            elif isinstance(value, (int, float)) and value in (0, 1):
                visual_input = bool(value)
            else:
                normalized = str(value).strip().lower()
                if normalized in {"0", "false", "no"}:
                    visual_input = False
                # The official parquet uses ``2`` for image-present Visual
                # Supplement/Dependent rows and ``1`` for the other image
                # rows; both are visual inputs.  The non-image split uses 0.
                elif normalized in {"1", "1.0", "2", "2.0", "true", "yes"}:
                    visual_input = True
                else:
                    raise ValueError(
                        f"invalid HallusionBench visual_input for {sample_id}: {value!r}"
                    )
            if visual_input:
                if "image" not in frame.columns:
                    raise ValueError(f"{path} has visual samples but no image column")
                image = decode_image(row["image"], sample_id)
            else:
                image = None
            details = required_text(row["gt_answer_details"], "gt_answer_details", sample_id)
            metadata = {
                "row_number": int(row_number),
                "category": str(row["category"]),
                "subcategory": str(row["subcategory"]),
                "set_id": str(row["set_id"]),
                "figure_id": str(row["figure_id"]),
                "question_id": str(row["question_id"]),
                "visual_input": visual_input,
            }
            yield BenchmarkSample(
                sample_id=sample_id,
                group_id=(
                    f"hallusionbench-{split}-{row['category']}-{row['subcategory']}-{row['set_id']}"
                ),
                dataset="hallusionbench",
                split=split,
                question=required_text(row["question"], "question", sample_id),
                references=(_yes_no(row["gt_answer"], sample_id), details),
                image=image,
                metadata=metadata,
            )
