"""No-op judge for deferred free-form labeling."""

from __future__ import annotations

from typing import Any

from .base import Judge


class NoneJudge(Judge):
    """Leave correctness unlabeled so a later dataset-specific judge can fill it."""

    def judge(self, prediction: str, gold: Any, sample: dict[str, Any] | None = None) -> None:
        return None
