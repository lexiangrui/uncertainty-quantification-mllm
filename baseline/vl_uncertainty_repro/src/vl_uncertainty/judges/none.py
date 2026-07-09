"""No-op judge."""

from __future__ import annotations

from typing import Any

from .base import Judge


class NoneJudge(Judge):
    def judge(self, prediction: str, gold: Any, sample: dict[str, Any] | None = None) -> bool | None:
        return None
