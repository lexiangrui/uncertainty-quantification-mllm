from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptFile:
    text: str
    sha256: str


def load_prompt(path: Path) -> PromptFile:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt file is empty: {path}")
    return PromptFile(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
