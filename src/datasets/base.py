from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class BenchmarkSample:
    """One question instance sent to a tested model."""

    sample_id: str
    group_id: str
    dataset: str
    split: str
    question: str
    references: tuple[str, ...]
    image: Image.Image | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.group_id:
            raise ValueError("sample_id and group_id must be non-empty")
        if not self.question.strip():
            raise ValueError(f"empty question for {self.sample_id}")
        if not self.references or any(not item.strip() for item in self.references):
            raise ValueError(f"invalid references for {self.sample_id}")
