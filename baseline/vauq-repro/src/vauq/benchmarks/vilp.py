"""ViLP benchmark loader.

The VAUQ paper uses the paired factual/counterfactual subset: two image-answer
cases per question, for 600 QIA triplets total.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

from .base import Benchmark


class VILP(Benchmark):
    DATASET_REPO = "ViLP/ViLP"
    PARQUET_FILE = "ViLP.parquet"

    def __init__(
        self,
        source: str = DATASET_REPO,
        parquet_file: str = PARQUET_FILE,
        cases: tuple[int, ...] = (1, 2),
    ):
        parquet_path = _resolve_vilp_parquet(source, parquet_file)
        self.ds = pd.read_parquet(parquet_path)
        self.cases = tuple(cases)

    def obtain_size(self) -> int:
        return len(self.ds) * len(self.cases)

    def retrieve(self, idx: int) -> dict | None:
        original_idx = idx // len(self.cases)
        case = self.cases[idx % len(self.cases)]
        row = self.ds.iloc[original_idx]
        question = (
            f"{row['question']}\n"
            "NOTE: Provide only the final answer. Do not provide unrelated details."
        )
        image = _decode_image(row.get(f"image{case}"))
        return {
            "idx": idx,
            "original_idx": int(original_idx),
            "case": int(case),
            "img": image,
            "question": question,
            "gt_ans": row.get(f"answer{case}"),
            "subset": f"case{case}",
        }


def _decode_image(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(BytesIO(value)).convert("RGB")
    return None


def _resolve_vilp_parquet(source: str, parquet_file: str) -> str:
    explicit = os.environ.get("VILP_PARQUET_PATH")
    candidates = [Path(explicit)] if explicit else []
    datasets_root = os.environ.get("VAUQ_DATASETS_DIR") or os.environ.get(
        "HF_DATASETS_CACHE"
    )
    if datasets_root:
        candidates.append(Path(datasets_root) / "vilp" / parquet_file)
    for path in candidates:
        if path.exists():
            return str(path)
    return hf_hub_download(
        repo_id=source,
        filename=parquet_file,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
