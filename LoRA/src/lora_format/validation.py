from __future__ import annotations

import re
from typing import Any


class ValidationError(ValueError):
    pass


FIELDS = ("vision", "reasoning", "answer")
MARKUP = re.compile(r"<[^>]*>|```|(?:^|\n)\s*#{1,6}\s")


def normalize_vqa_answer(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _validate_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string")
    value = " ".join(value.strip().split())
    if not value:
        raise ValidationError(f"{name} is empty")
    if MARKUP.search(value):
        raise ValidationError(f"{name} contains markup")
    return value


def validate_teacher_payload(payload: Any, expected_answer: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValidationError("teacher output must be a JSON object")
    if set(payload) != set(FIELDS):
        raise ValidationError(f"teacher output must contain exactly {FIELDS}")
    clean = {
        "vision": _validate_text("vision", payload["vision"]),
        "reasoning": _validate_text("reasoning", payload["reasoning"]),
        "answer": _validate_text("answer", payload["answer"]),
    }
    if normalize_vqa_answer(clean["answer"]) != normalize_vqa_answer(expected_answer):
        raise ValidationError("generated answer differs from the VQAv2 majority answer")
    return clean
