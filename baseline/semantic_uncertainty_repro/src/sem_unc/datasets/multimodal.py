"""Multimodal QA dataset implementations.

Mirrors vauq-repro's benchmark implementations for CV-Bench, MMVet, and
ViLP.  Each class conforms to the ``Dataset`` ABC.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from .base import Dataset


# ------------------------------------------------------------------
# CV-Bench
# ------------------------------------------------------------------
class CVBenchDataset(Dataset):
    """CV-Bench (2D + 3D concatenated, 2638 samples, multiple-choice)."""

    def __init__(
        self,
        source: str = "nyu-visionx/CV-Bench",
        configs: tuple[str, ...] = ("2D", "3D"),
        split: str = "test",
    ):
        parts = [load_dataset(source, c)[split] for c in configs]
        self._ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        row = self._ds[idx]
        question = (
            f"{row['prompt']}\n"
            "Your answer should be only the letter of the option."
        )
        return {
            "id": f"cvbench-{idx}",
            "question": question,
            "img": row["image"],
            "gt_ans": row["answer"],
            "choices": row.get("choices"),
            "subset": row.get("type"),
        }


# ------------------------------------------------------------------
# MM-Vet
# ------------------------------------------------------------------
class MMVetDataset(Dataset):
    """MM-Vet (free-form VQA)."""

    def __init__(self, source: str = "whyu/mm-vet", split: str = "test"):
        self._ds = load_dataset(source)
        self._split = split

    def __len__(self) -> int:
        return len(self._ds[self._split])

    def __getitem__(self, idx: int) -> dict:
        row = self._ds[self._split][idx]
        question = (
            f"{row['question']}\n"
            "NOTE: Provide only the final answer. Do not provide unrelated details."
        )
        return {
            "id": f"mmvet-{idx}",
            "question": question,
            "img": row["image"],
            "gt_ans": row["answer"],
        }


# ------------------------------------------------------------------
# ViLP
# ------------------------------------------------------------------
class VILPDataset(Dataset):
    """ViLP paired factual / counterfactual subset (600 QIA × 2 cases)."""

    DATASET_REPO = "ViLP/ViLP"
    PARQUET_FILE = "ViLP.parquet"

    def __init__(
        self,
        source: str = DATASET_REPO,
        parquet_file: str = PARQUET_FILE,
        cases: tuple[int, ...] = (1, 2),
    ):
        parquet_path = _resolve_vilp_parquet(source, parquet_file)
        self._ds = pd.read_parquet(parquet_path)
        self._cases = tuple(cases)

    def __len__(self) -> int:
        return len(self._ds) * len(self._cases)

    def __getitem__(self, idx: int) -> dict:
        original_idx = idx // len(self._cases)
        case = self._cases[idx % len(self._cases)]
        row = self._ds.iloc[original_idx]
        question = (
            f"{row['question']}\n"
            "NOTE: Provide only the final answer. Do not provide unrelated details."
        )
        image = _decode_image(row.get(f"image{case}"))
        return {
            "id": f"vilp-{original_idx}-case{case}",
            "question": question,
            "img": image,
            "gt_ans": row.get(f"answer{case}"),
            "subset": f"case{case}",
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
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
