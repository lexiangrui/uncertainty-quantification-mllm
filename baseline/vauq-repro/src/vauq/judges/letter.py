"""Multiple-choice letter judge: extract the option letter via regex.

For benchmarks whose gold answer is an option letter (CV-Bench). The model is
prompted to emit only the letter, but real outputs vary ("B", "The answer is B.",
"(C)", "A) ..."). We extract the first plausible letter and compare it to the
normalized gold letter.
"""

from __future__ import annotations

import re
from typing import Any

from .base import Judge

# Ordered regexes; each captured letter must not be immediately followed by
# another letter (negative lookahead) so words like "BLUE" don't yield "B".
_PATTERNS = [
    re.compile(r"(?i)(?:answer|option|choice)\s*(?:is|:)?\s*\(?([a-z])\)?(?![a-z])"),
    re.compile(r"(?i)^\s*\(?([a-z])\)?(?:[\.\):,]\s*|$)"),
    re.compile(r"(?i)\b([a-z])\b(?![a-z])"),
]


def _extract_letter(text: str, max_letter: str = "Z") -> str | None:
    if not text:
        return None
    for rx in _PATTERNS:
        m = rx.search(text)
        if m:
            letter = m.group(1).upper()
            if "A" <= letter <= max_letter.upper():
                return letter
    caps = re.findall(r"[A-Z]", text)
    if caps and caps[0] <= max_letter.upper():
        return caps[0]
    return None


def _normalize_gold(gold: Any, choices: list | None) -> str | None:
    g = str(gold).strip()
    if len(g) == 1 and g.isalpha():
        return g.upper()
    if choices:
        gn = g.lower()
        for i, ch in enumerate(choices):
            if str(ch).strip().lower() == gn:
                return chr(ord("A") + i)
    m = re.match(r"\s*\(?([A-Za-z])\)?", g)
    return m.group(1).upper() if m else None


class LetterJudge(Judge):
    def judge(self, prediction: str, gold: Any, sample: dict[str, Any] | None = None) -> bool:
        sample = sample or {}
        choices = sample.get("choices") or []
        max_letter = chr(ord("A") + max(0, len(choices) - 1)) if choices else "Z"
        gold_letter = _normalize_gold(gold, choices)
        pred_letter = _extract_letter(prediction or "", max_letter=max_letter)
        return pred_letter is not None and gold_letter is not None and pred_letter == gold_letter
