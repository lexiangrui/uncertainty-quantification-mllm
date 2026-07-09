"""Text-model judge for free-form answers."""

from __future__ import annotations

import re
from typing import Any

from vl_uncertainty.text_models import TextModel

from .base import Judge


class TextModelJudge(Judge):
    def __init__(self, text_model: TextModel):
        self.text_model = text_model
        self.last_result: dict[str, Any] | None = None

    def judge(self, prediction: str, gold: Any, sample: dict[str, Any] | None = None) -> bool | None:
        prompt = (
            f"Ground truth: {gold}. Model answer: {prediction}. "
            "Please verify if the model answer matches the ground truth. "
            "Respond with either 'Correct' or 'Wrong' only."
        )
        response = self.text_model.generate(prompt, temp=0.1, max_new_tokens=8)
        correct = _parse_correct(response)
        self.last_result = {"raw_response": response, "correct": correct}
        return correct


def _parse_correct(text: str) -> bool:
    match = re.search(r"\b(correct|wrong|incorrect)\b", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).lower() == "correct"
    return "correct" in text.lower()
