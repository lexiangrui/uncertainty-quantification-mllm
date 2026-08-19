from lora_format.format_evaluation import evaluate_response, normalize_vqa
from lora_format.xml import build_xml_response


VALID = "<vision>Two dogs are visible.</vision><reasoning>Counting them gives two.</reasoning><answer>two</answer>"


def test_strict_xml_and_vqa_answer() -> None:
    result = evaluate_response(VALID, "2")
    assert result["strict_xml_valid"] is True
    assert result["answer_correct"] is True
    assert result["parsed"]["vision"] == "Two dogs are visible."


def test_xml_tags_are_case_insensitive_like_generation_parser() -> None:
    result = evaluate_response(VALID.replace("vision", "Vision").replace("reasoning", "Reasoning").replace("answer", "Answer"), "2")
    assert result["strict_xml_valid"] is True
    assert result["answer_correct"] is True


def test_extra_text_and_repeated_tag_fail_strict_validation() -> None:
    extra = evaluate_response("Here: " + VALID, "2")
    assert extra["strict_xml_valid"] is False
    assert extra["extra_text"] is True
    repeated = evaluate_response(VALID + "\n<answer>2</answer>", "2")
    assert repeated["tags_once"] is False
    assert repeated["strict_xml_valid"] is False


def test_missing_or_reordered_sections_are_detected() -> None:
    response = "<answer>yes</answer><vision>A cat.</vision><reasoning>It is visible.</reasoning>"
    result = evaluate_response(response, "yes")
    assert result["tag_order_correct"] is False
    assert result["strict_xml_valid"] is False


def test_vqa_normalization() -> None:
    assert normalize_vqa("The TWO.") == normalize_vqa("2")


def test_xml_escape_round_trip_preserves_ampersand_answer() -> None:
    response = build_xml_response("A sign is visible.", "Read the sign.", "ben & jerry's")
    result = evaluate_response(response, "ben & jerry's")
    assert result["strict_xml_valid"] is True
    assert result["parsed"]["answer"] == "ben & jerry's"
    assert result["answer_correct"] is True
