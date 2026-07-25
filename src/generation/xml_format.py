from __future__ import annotations


XML_ZERO_SHOT_INSTRUCTION = """Answer using exactly these three XML tags once and
in order, with no line breaks and no text outside them:
<vision>relevant visible evidence</vision><reasoning>brief reasoning</reasoning><answer>concise final answer</answer>"""


def xml_response(vision: str, reasoning: str, answer: str) -> str:
    values = (vision.strip(), reasoning.strip(), answer.strip())
    if any(not value for value in values):
        raise ValueError("XML response sections must be non-empty")
    if any("<" in value or ">" in value for value in values):
        raise ValueError("XML response section content cannot contain tags")
    return (
        f"<vision>{values[0]}</vision>"
        f"<reasoning>{values[1]}</reasoning>"
        f"<answer>{values[2]}</answer>"
    )
