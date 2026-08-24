from __future__ import annotations

import re
from pathlib import Path


def judge_directory_name(model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.strip().lower()).strip("_")
    if not slug:
        raise ValueError("judge model cannot produce an empty directory name")
    return f"judging_{slug}"


def validate_raw_judge_output(output: Path, model: str) -> None:
    expected = judge_directory_name(model)
    if expected not in output.parts:
        raise ValueError(
            f"raw judge output for {model!r} must be under a directory named "
            f"{expected!r}, not {output}"
        )
