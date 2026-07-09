"""VisualCoT benchmark loader.

The public Visual-CoT dataset stores metadata separately from image tar shards.
This loader reads the GQA validation metadata used as an evaluation split and
loads images from an extracted image directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

from .base import Benchmark


class VisualCoT(Benchmark):
    DATASET_REPO = "deepcs233/Visual-CoT"
    METADATA_FILE = "cot_with_detailed_reasoning_steps/gqa_cot_val.jsonl"

    def __init__(
        self,
        source: str = DATASET_REPO,
        metadata_file: str = METADATA_FILE,
        image_dir: str | None = None,
    ):
        metadata_path = _resolve_visualcot_metadata(source, metadata_file)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

        default_image_dir = os.environ.get("VISUALCOT_IMAGE_DIR")
        if not default_image_dir:
            datasets_root = os.environ.get("VAUQ_DATASETS_DIR") or os.environ.get(
                "HF_DATASETS_CACHE"
            )
            if datasets_root:
                default_image_dir = str(Path(datasets_root) / "visualcot_images")
        image_dir_value = image_dir or default_image_dir
        self.image_dir = Path(image_dir_value) if image_dir_value else None
        self._image_index: dict[str, Path] | None = None

    def obtain_size(self) -> int:
        return len(self.rows)

    def retrieve(self, idx: int) -> dict | None:
        row = self.rows[idx]
        image = self._load_image(row.get("image"))
        question = (
            f"{row['question']}\n"
            "NOTE: Provide only the final answer. Do not provide unrelated details."
        )
        return {
            "idx": idx,
            "img": image,
            "question": question,
            "gt_ans": row.get("answer") or row.get("full_answer"),
            "subset": row.get("dataset") or row.get("split"),
            "bboxs": row.get("bboxs"),
        }

    def _load_image(self, filename: str | None):
        if not filename:
            return None
        path = self._resolve_image_path(filename)
        if path is None:
            return None
        return Image.open(path).convert("RGB")

    def _resolve_image_path(self, filename: str) -> Path | None:
        if self.image_dir is None:
            return None
        direct = self.image_dir / filename
        if direct.exists():
            return direct
        if self._image_index is None:
            self._image_index = {
                path.name: path for path in self.image_dir.rglob("*") if path.is_file()
            }
        return self._image_index.get(filename)


def _resolve_visualcot_metadata(source: str, metadata_file: str) -> str:
    explicit = os.environ.get("VISUALCOT_METADATA_PATH")
    candidates = [Path(explicit)] if explicit else []
    datasets_root = os.environ.get("VAUQ_DATASETS_DIR") or os.environ.get(
        "HF_DATASETS_CACHE"
    )
    if datasets_root:
        candidates.append(Path(datasets_root) / "visualcot" / metadata_file)
    for path in candidates:
        if path.exists():
            return str(path)
    return hf_hub_download(
        repo_id=source,
        filename=metadata_file,
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
