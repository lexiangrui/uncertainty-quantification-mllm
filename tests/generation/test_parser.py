import pytest

from src.generation.parser import answer_character_span, parse_structured_response


VALID = (
    "<vision>A red ball is visible.</vision>"
    "<reasoning>The visible object determines the answer.</reasoning>"
    "<answer>red</answer>"
)

PLAIN_VALID = (
    "Visual Observation: A red ball is visible.\n"
    "Reasoning: The visible object determines the answer.\n"
    "Final Answer: red"
)


def test_parse_structured_response() -> None:
    parsed = parse_structured_response(VALID)
    assert parsed.vision == "A red ball is visible."
    assert parsed.reasoning == "The visible object determines the answer."
    assert parsed.answer == "red"
    start, end = answer_character_span(VALID)
    assert VALID[start:end] == "red"


def test_parse_plain_three_part_response() -> None:
    parsed = parse_structured_response(PLAIN_VALID, "plain_sections")
    assert parsed.vision == "A red ball is visible."
    assert parsed.reasoning == "The visible object determines the answer."
    assert parsed.answer == "red"
    start, end = answer_character_span(PLAIN_VALID, "plain_sections")
    assert PLAIN_VALID[start:end] == "red"


def test_parse_plain_three_part_response_accepts_bold_labels() -> None:
    value = PLAIN_VALID.replace("Visual Observation", "**Visual Observation**").replace(
        "Reasoning", "**Reasoning**"
    ).replace("Final Answer", "**Final Answer**")
    assert parse_structured_response(value, "plain_sections").answer == "red"


@pytest.mark.parametrize(
    "response",
    [
        VALID.replace("<answer>", ""),
        VALID.replace("red", ""),
        VALID.replace("<vision>", "<answer>", 1),
    ],
)
def test_reject_non_contract_response(response: str) -> None:
    with pytest.raises(ValueError):
        parse_structured_response(response)
