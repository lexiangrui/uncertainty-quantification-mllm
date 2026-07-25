import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm_judge import (
    MMHAL_SYSTEM_PROMPT,
    RegexChoiceJudge,
    build_mmhal_messages,
    build_judge_prompt,
    extract_choice,
    extract_choice_letter,
    extract_yes_no,
    parse_judge_verdict,
    parse_mmhal_response,
)


def test_letter_choice_formats():
    judge = RegexChoiceJudge()
    choices = ["red", "green", "blue", "black"]
    assert judge.judge("C", 2, choices)
    assert judge.judge("(C) blue", 2, choices)
    assert judge.judge("The answer is C.", 2, choices)
    assert not judge.judge("blue", 2, choices)


def test_number_choice_formats_and_range():
    choices = ["red", "green", "blue", "black"]
    assert extract_choice("2", 4, "number") == 2
    assert extract_choice("(2): blue", 4, "number") == 2
    assert extract_choice("8", 4, "number") is None


def test_extract_choice_letter_returns_letter_or_none():
    assert extract_choice_letter("(B)", 4) == "B"
    assert extract_choice_letter("The answer is C.", 4) == "C"
    assert extract_choice_letter("blue", 4) is None
    # 不把句子里的 A/B/C/D（如 "answer"）误判为选项
    assert extract_choice_letter("The answer is B", 4) == "B"


def test_yes_no_formats_are_strict():
    judge = RegexChoiceJudge()
    choices = ["No", "Yes"]
    assert extract_yes_no("Yes") == 1
    assert extract_yes_no("The answer is no.") == 0
    assert extract_yes_no("Yesterday") is None
    assert judge.judge("YES", 1, choices, mode="yes_no")
    assert judge.judge("No.", 0, choices, mode="yes_no")


def test_strict_llm_verdict():
    assert parse_judge_verdict('{"verdict":"CORRECT"}')
    assert not parse_judge_verdict('{"verdict":"WRONG"}')
    prompt = json.loads(build_judge_prompt("Where?", ["left"], "on the left"))
    assert prompt["question"] == "Where?"
    assert prompt["reference_answers"] == ["left"]


def test_mmhal_response_round_trip_and_hallucination_rule():
    parsed = parse_mmhal_response('{"analysis":"ok","correct":true,"rating":4}')
    assert parsed == {"analysis": "ok", "correct": True, "rating": 4, "hallucination": False}
    hallucinated = parse_mmhal_response('{"analysis":"bad","correct":false,"rating":1}')
    assert hallucinated["hallucination"] is True


def test_mmhal_response_rejects_malformed():
    for bad in [
        '{"analysis":"ok","correct":true}',            # missing rating
        '{"analysis":"ok","correct":"true","rating":4}',  # correct not bool
        '{"analysis":"ok","correct":true,"rating":7}',   # rating out of range
        'not json',
        '{"analysis":"","correct":true,"rating":4}',     # empty analysis
    ]:
        try:
            parse_mmhal_response(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_mmhal_messages_shape():
    messages = build_mmhal_messages("mmvet", "<img>", "Q?", "ref", "pred")
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "text"
    assert "Q?" in content[1]["text"] and "pred" in content[1]["text"]
    assert "MMHal-Bench rating" in messages[0]["content"][0]["text"]


def test_mmhal_system_prompt_is_stable():
    # 防止不经意改动 prompt（下游用其 sha256 做版本指纹）
    assert "official MMHal-Bench ratings" in MMHAL_SYSTEM_PROMPT
    assert len(MMHAL_SYSTEM_PROMPT) > 1000


if __name__ == "__main__":
    test_letter_choice_formats()
    test_number_choice_formats_and_range()
    test_extract_choice_letter_returns_letter_or_none()
    test_strict_llm_verdict()
    test_mmhal_response_round_trip_and_hallucination_rule()
    test_mmhal_response_rejects_malformed()
    test_mmhal_messages_shape()
    test_mmhal_system_prompt_is_stable()
    print("judge tests passed")
