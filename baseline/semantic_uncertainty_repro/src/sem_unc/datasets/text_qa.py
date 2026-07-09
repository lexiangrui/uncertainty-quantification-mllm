"""Text QA datasets (TriviaQA, SQuAD, NQ, SVAMP)."""

from __future__ import annotations

import hashlib
import json
import logging
import os

import datasets as hf_datasets

from .base import Dataset

logger = logging.getLogger(__name__)


class TriviaQADataset(Dataset):
    """TriviaQA in SQuAD format (auto-downloaded from HuggingFace)."""

    def __init__(
        self,
        source: str = "TimoImhof/TriviaQA-in-SQuAD-format",
        split: str = "validation",
        seed: int = 10,
    ):
        ds = hf_datasets.load_dataset(source)["unmodified"]
        split_ds = ds.train_test_split(test_size=0.2, seed=seed)
        self._data = split_ds["test" if split == "validation" else "train"]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        row = self._data[idx]
        # TriviaQA-in-SQuAD-format has: id, title, context, question, answers
        answers = row["answers"]["text"]
        return {
            "id": row.get("id", f"triviaqa-{idx}"),
            "question": row["question"],
            "img": None,
            "gt_ans": answers[0] if answers else "",
            "all_gt_ans": answers,
            "context": row.get("context", ""),
        }

class SQuADDataset(Dataset):
    """SQuAD v2 (includes unanswerable questions)."""

    def __init__(
        self,
        source: str = "squad_v2",
        split: str = "validation",
    ):
        ds = hf_datasets.load_dataset(source)
        self._data = ds[split]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        row = self._data[idx]
        answers = row["answers"]["text"]
        return {
            "id": row.get("id", f"squad-{idx}"),
            "question": row["question"],
            "img": None,
            "gt_ans": answers[0] if answers else "",
            "all_gt_ans": answers,
            "context": row.get("context", ""),
        }


class NaturalQuestionsDataset(Dataset):
    """Natural Questions (open-domain)."""

    def __init__(
        self,
        source: str = "nq_open",
        split: str = "validation",
    ):
        ds = hf_datasets.load_dataset(source)
        self._data = ds[split]
        self._md5hash = lambda s: str(
            int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
        )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        row = self._data[idx]
        return {
            "id": self._md5hash(str(row["question"])),
            "question": row["question"] + "?",
            "img": None,
            "gt_ans": row["answer"][0] if row["answer"] else "",
            "all_gt_ans": row["answer"],
            "context": "",
        }


class SVAMPDataset(Dataset):
    """SVAMP math word problems."""

    def __init__(
        self,
        source: str = "ChilleD/SVAMP",
        split: str = "test",
    ):
        ds = hf_datasets.load_dataset(source)
        self._data = [
            {
                "id": d["ID"],
                "question": d["Question"],
                "img": None,
                "gt_ans": str(d["Answer"]),
                "all_gt_ans": [str(d["Answer"])],
                "context": d.get("Body", ""),
            }
            for d in ds["test" if split == "validation" else "train"]
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]
