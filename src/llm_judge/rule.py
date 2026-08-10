"""Deterministic regex judge for closed multiple-choice answers."""

from __future__ import annotations

import re
from typing import Any


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_yes_no(text: Any) -> int | None:
    """Return 0 for No and 1 for Yes from a short deterministic answer."""
    value = str(text or "").strip()
    patterns = (
        r"^(?:THE\s+)?ANSWER(?:\s+IS|\s*:)?\s*(YES|NO)[.!,:;]?(?:\s+.*)?$",
        r"^(YES|NO)[.!,:;]?(?:\s+.*)?$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, value, flags=re.IGNORECASE)
        if match:
            return 1 if match.group(1).upper() == "YES" else 0
    return None


def extract_choice(text: Any, num_choices: int, mode: str) -> int | None:
    """Return a zero-based option index from a model answer."""
    if not 2 <= num_choices <= len(LETTERS):
        raise ValueError(f"invalid number of choices: {num_choices}")
    value = str(text or "").strip()
    if mode == "letter":
        allowed = LETTERS[:num_choices]
        patterns = (
            rf"^(?:THE\s+)?(?:ANSWER|OPTION|CHOICE)(?:\s+IS|\s*:)?\s*[\(\[]?([{allowed}])[\)\].:,]?(?:\s+.*)?$",
            rf"^[\(\[]?([{allowed}])[\)\].:,]?(?:\s+.*)?$",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, value, flags=re.IGNORECASE)
            if match:
                return LETTERS.index(match.group(1).upper())
        return None
    if mode == "number":
        patterns = (
            r"^(?:THE\s+)?(?:ANSWER|OPTION|CHOICE)(?:\s+IS|\s*:)?\s*\(?([0-9]+)\)?(?:\s*[:).,-]\s*.*)?$",
            r"^\(?([0-9]+)\)?(?:\s*[:).,-]\s*.*)?$",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, value, flags=re.IGNORECASE)
            if match:
                index = int(match.group(1))
                return index if 0 <= index < num_choices else None
        return None
    raise ValueError(f"unknown choice mode: {mode!r}")


def extract_choice_letter(text: Any, num_choices: int) -> str | None:
    """Return the parsed option letter (e.g. ``"B"``) for a letter-mode answer.

    Convenience wrapper around :func:`extract_choice` for callers that compare
    or display letters instead of zero-based indices.
    """
    index = extract_choice(text, num_choices, "letter")
    return LETTERS[index] if index is not None else None


class RuleJudge:
    """Compare a parsed prediction with a zero-based gold option index."""

    name = "rule"

    def judge(
        self,
        prediction: str,
        gold_index: int,
        choices: list[Any],
        mode: str = "letter",
    ) -> bool:
        if mode == "yes_no":
            if len(choices) != 2 or int(gold_index) not in {0, 1}:
                raise ValueError("yes/no judge requires choices [No, Yes] and gold 0/1")
            return extract_yes_no(prediction) == int(gold_index)
        if not 0 <= int(gold_index) < len(choices):
            raise ValueError(f"gold index {gold_index} outside {len(choices)} choices")
        return extract_choice(prediction, len(choices), mode) == int(gold_index)
