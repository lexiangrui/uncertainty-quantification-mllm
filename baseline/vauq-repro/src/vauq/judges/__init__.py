"""Judge registry."""

from __future__ import annotations

from .base import Judge
from .letter import LetterJudge
from .llm import LLMJudge
from .none import NoneJudge
from .qwen_local import QwenLocalJudge

JUDGE_MAP: dict[str, type[Judge]] = {
    "letter": LetterJudge,
    "llm": LLMJudge,
    "none": NoneJudge,
    "qwen_local": QwenLocalJudge,
}

# Per-benchmark default judge.
DEFAULT_JUDGE: dict[str, str] = {
    "cvbench": "letter",
    "mmvet": "qwen_local",
    "vilp": "qwen_local",
}


def build_judge(name: str | None, benchmark: str | None = None) -> Judge:
    if name is None:
        name = DEFAULT_JUDGE.get(benchmark or "", "none")
    if name not in JUDGE_MAP:
        raise ValueError(f"Unknown judge {name!r}; choose from {sorted(JUDGE_MAP)}")
    return JUDGE_MAP[name]()


__all__ = [
    "Judge",
    "LetterJudge",
    "LLMJudge",
    "NoneJudge",
    "QwenLocalJudge",
    "JUDGE_MAP",
    "DEFAULT_JUDGE",
    "build_judge",
]
