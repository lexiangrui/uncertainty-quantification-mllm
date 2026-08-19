from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ._image import decode_image, require_columns, required_text
from .base import BenchmarkSample


def iter_mmvet(path: Path) -> Iterator[BenchmarkSample]:
    if not path.is_file():
        raise FileNotFoundError(path)
    import pandas as pd

    frame = pd.read_parquet(path)
    require_columns(
        list(frame.columns), {"question_id", "question", "answer", "image"}, str(path)
    )
    for row_number, row in frame.iterrows():
        source_id = required_text(row["question_id"], "question_id", f"mmvet-row-{row_number}")
        sample_id = f"mmvet-{source_id}"
        metadata = {"row_number": int(row_number), "source_id": source_id}
        if "capability" in frame.columns:
            metadata["capability"] = row["capability"]
        if "image_source" in frame.columns:
            metadata["image_source"] = row["image_source"]
        yield BenchmarkSample(
            sample_id=sample_id,
            group_id=sample_id,
            dataset="mmvet",
            split="test",
            question=required_text(row["question"], "question", sample_id),
            references=(required_text(row["answer"], "answer", sample_id),),
            image=decode_image(row["image"], sample_id),
            metadata=metadata,
        )
