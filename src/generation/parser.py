from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StructuredResponse:
    vision: str
    reasoning: str
    answer: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


XML_SECTION_PATTERN = re.compile(
    r"^\s*<vision>\s*(?P<vision>.+?)\s*</vision>\s*"
    r"<reasoning>\s*(?P<reasoning>.+?)\s*</reasoning>\s*"
    r"<answer>\s*(?P<answer>.+?)\s*</answer>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _pattern(response_format: str) -> re.Pattern:
    if response_format == "xml":
        return XML_SECTION_PATTERN
    raise ValueError(f"unknown response format: {response_format}")


def parse_structured_response(
    text: str, response_format: str = "xml"
) -> StructuredResponse:
    match = _pattern(response_format).fullmatch(text)
    if match is None:
        raise ValueError(
            f"response cannot be separated using {response_format} format"
        )
    values = {
        name: match.group(name).strip() for name in ("vision", "reasoning", "answer")
    }
    empty = [name for name, value in values.items() if not value]
    if empty:
        raise ValueError(f"response section is empty: {empty[0]}")
    return StructuredResponse(
        vision=values["vision"],
        reasoning=values["reasoning"],
        answer=values["answer"],
    )


def answer_character_span(
    text: str, response_format: str = "xml"
) -> tuple[int, int]:
    match = _pattern(response_format).fullmatch(text)
    if match is None or not match.group("answer").strip():
        raise ValueError("response has no separable answer section")
    start, end = match.span("answer")
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
