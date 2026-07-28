import pytest

from lora_format.validation import ValidationError, validate_teacher_payload
from lora_format.xml import build_xml_response


def test_valid_payload_is_wrapped_by_program() -> None:
    clean = validate_teacher_payload(
        {
            "vision": "A blue mug is on the table.",
            "reasoning": "The visible mug is blue, so its color is blue.",
            "answer": "blue",
        },
        "blue",
    )
    assert build_xml_response(**clean) == (
        "<vision>A blue mug is on the table.</vision>"
        "<reasoning>The visible mug is blue, so its color is blue.</reasoning>"
        "<answer>blue</answer>"
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"vision": "x", "reasoning": "y", "answer": "red"}, "differs"),
        ({"vision": "<vision>x</vision>", "reasoning": "y", "answer": "blue"}, "markup"),
        ({"vision": "x", "reasoning": "y", "answer": "blue", "extra": "z"}, "exactly"),
    ],
)
def test_invalid_teacher_payload_is_rejected(payload: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        validate_teacher_payload(payload, "blue")


def test_long_grounded_text_is_not_rejected() -> None:
    vision = "visible detail " * 100
    reasoning = "grounded inference " * 100
    clean = validate_teacher_payload(
        {"vision": vision, "reasoning": reasoning, "answer": "blue"},
        "blue",
    )
    assert clean["vision"] == vision.strip()
    assert clean["reasoning"] == reasoning.strip()


def test_supervision_metadata_text_is_not_filtered() -> None:
    clean = validate_teacher_payload(
        {
            "vision": "The ground truth says stop.",
            "reasoning": "Following the provided instruction, the answer is stop.",
            "answer": "stop",
        },
        "stop",
    )
    assert clean["answer"] == "stop"


def test_xml_escapes_text() -> None:
    assert "A &amp; B" in build_xml_response("A & B", "A < B", "yes")
    assert "A &lt; B" in build_xml_response("A & B", "A < B", "yes")
