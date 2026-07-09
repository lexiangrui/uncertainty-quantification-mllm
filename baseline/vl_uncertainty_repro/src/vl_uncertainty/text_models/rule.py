"""Rule-based text helper for cheap smoke tests."""

from __future__ import annotations

from .base import TextModel


class EchoTextModel(TextModel):
    """Return deterministic prompt-derived outputs without loading an LLM."""

    def generate(self, prompt: str, temp: float = 0.1, max_new_tokens: int = 256) -> str:
        lower = prompt.lower()
        if "respond with either 'yes' or 'no'" in lower or 'respond with either "yes" or "no"' in lower:
            return "Yes" if _quoted_pair_overlap(prompt) else "No"
        if "respond with either 'correct' or 'wrong'" in lower or 'respond with either "correct" or "wrong"' in lower:
            return "Correct" if _quoted_pair_overlap(prompt) else "Wrong"
        if "given the input question:" in lower:
            return _extract_between(prompt, "Given the input question: '", "'") or prompt
        return prompt[:max_new_tokens]


def _quoted_pair_overlap(text: str) -> bool:
    quoted = []
    parts = text.split("'")
    for i in range(1, len(parts), 2):
        quoted.append(parts[i].lower())
    if len(quoted) < 2:
        return True
    a = set(quoted[0].split())
    b = set(quoted[1].split())
    if not a or not b:
        return quoted[0].strip() == quoted[1].strip()
    return bool(a & b)


def _extract_between(text: str, start: str, end: str) -> str | None:
    begin = text.find(start)
    if begin < 0:
        return None
    begin += len(start)
    finish = text.find(end, begin)
    if finish < 0:
        return None
    return text[begin:finish]
