"""Judge registry."""

from __future__ import annotations

from vl_uncertainty.text_models import TextModel

from .base import Judge
from .choice import ChoiceJudge
from .llm import TextModelJudge
from .none import NoneJudge


def build_judge(name: str, text_model: TextModel | None = None) -> Judge:
    if name == "choice":
        return ChoiceJudge()
    if name == "llm":
        if text_model is None:
            raise ValueError("text_model is required for the llm judge")
        return TextModelJudge(text_model)
    if name == "none":
        return NoneJudge()
    raise ValueError("Unknown judge {!r}; choose from ['choice', 'llm', 'none']".format(name))


__all__ = ["Judge", "ChoiceJudge", "TextModelJudge", "NoneJudge", "build_judge"]
