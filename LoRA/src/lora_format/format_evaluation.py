from __future__ import annotations

import re
import string
from typing import Any
from xml.sax.saxutils import unescape


TAGS = ("vision", "reasoning", "answer")
STRICT_XML = re.compile(
    r"\A\s*<vision>\s*(?P<vision>.*?)\s*</vision>\s*"
    r"<reasoning>\s*(?P<reasoning>.*?)\s*</reasoning>\s*"
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*\Z",
    re.DOTALL,
)
ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_vqa(value: str) -> str:
    text = value.lower().replace("\n", " ").replace("\t", " ")
    text = text.translate(str.maketrans({character: " " for character in string.punctuation}))
    words = []
    for word in text.split():
        if word in ARTICLES:
            continue
        words.append(NUMBER_WORDS.get(word, word))
    return " ".join(words)


def evaluate_response(response: str, expected_answer: str) -> dict[str, Any]:
    counts = {tag: response.count(f"<{tag}>") + response.count(f"</{tag}>") for tag in TAGS}
    tags_once = all(counts[tag] == 2 for tag in TAGS)
    opening_positions = [response.find(f"<{tag}>") for tag in TAGS]
    tag_order_correct = all(position >= 0 for position in opening_positions) and opening_positions == sorted(opening_positions)
    match = STRICT_XML.fullmatch(response)
    parsed = (
        {tag: unescape(match.group(tag).strip()) for tag in TAGS}
        if match
        else None
    )
    nonempty_sections = parsed is not None and all(parsed[tag] for tag in TAGS)
    strict_xml_valid = bool(match and tags_once and nonempty_sections)
    predicted_answer = parsed["answer"] if parsed else None
    answer_correct = bool(
        predicted_answer is not None
        and normalize_vqa(predicted_answer) == normalize_vqa(expected_answer)
    )
    stripped = response.strip()
    extra_text = not (
        stripped.startswith("<vision>") and stripped.endswith("</answer>")
    )
    return {
        "strict_xml_valid": strict_xml_valid,
        "tag_order_correct": tag_order_correct,
        "tags_once": tags_once,
        "extra_text": extra_text,
        "nonempty_sections": nonempty_sections,
        "parsed": parsed,
        "predicted_answer": predicted_answer,
        "answer_correct": answer_correct,
    }
