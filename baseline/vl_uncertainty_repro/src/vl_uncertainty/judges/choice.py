"""Choice-answer judge."""

from __future__ import annotations

import re
from typing import Any

from .base import Judge


class ChoiceJudge(Judge):
    def judge(self, prediction: str, gold: Any, sample: dict[str, Any] | None = None) -> bool | None:
        match = re.search(r"\d+", prediction or "")
        if match is None:
            return False
        try:
            return int(match.group()) == int(gold)
        except (TypeError, ValueError):
            return False
