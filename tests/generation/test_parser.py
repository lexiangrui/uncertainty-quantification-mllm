import pytest

from src.generation.parser import answer_character_span, parse_structured_response


VALID = (
    "<vision>A red ball is visible.</vision>"
    "<reasoning>The visible object determines the answer.</reasoning>"
    "<answer>red</answer>"
)


def test_parse_structured_response() -> None:
    parsed = parse_structured_response(VALID)
    assert parsed.vision == "A red ball is visible."
    assert parsed.reasoning == "The visible object determines the answer."
    assert parsed.answer == "red"
    start, end = answer_character_span(VALID)
    assert VALID[start:end] == "red"


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
