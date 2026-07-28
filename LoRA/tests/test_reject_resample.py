from lora_format.reject_resample import (
    collect_valid_sample,
    package_reject_resample_result,
)


def _attempt(valid: bool, text: str = "x") -> dict:
    return {"strict_xml_valid": valid, "raw_response": text}


def test_accepts_first_valid_attempt() -> None:
    result = package_reject_resample_result(
        [_attempt(False, "bad"), _attempt(True, "good"), _attempt(True, "later")],
        max_attempts=10,
    )
    assert result["raw_response"] == "good"
    assert result["reject_resample"] == {
        "max_attempts": 10,
        "attempts_used": 2,
        "accepted": True,
        "rejected_count": 1,
    }


def test_falls_back_to_last_attempt_when_all_invalid() -> None:
    result = package_reject_resample_result(
        [_attempt(False, "a"), _attempt(False, "b"), _attempt(False, "c")],
        max_attempts=3,
    )
    assert result["raw_response"] == "c"
    assert result["reject_resample"] == {
        "max_attempts": 3,
        "attempts_used": 3,
        "accepted": False,
        "rejected_count": 3,
    }


def test_collect_valid_sample_stops_early() -> None:
    calls: list[int] = []

    def generate(attempt_index: int) -> dict:
        calls.append(attempt_index)
        return _attempt(attempt_index == 3, text=f"try-{attempt_index}")

    result = collect_valid_sample(generate, max_attempts=10)
    assert calls == [1, 2, 3]
    assert result["raw_response"] == "try-3"
    assert result["reject_resample"]["accepted"] is True
    assert result["reject_resample"]["attempts_used"] == 3
    assert result["reject_resample"]["rejected_count"] == 2


def test_collect_valid_sample_exhausts_budget() -> None:
    calls: list[int] = []

    def generate(attempt_index: int) -> dict:
        calls.append(attempt_index)
        return _attempt(False, text=f"try-{attempt_index}")

    result = collect_valid_sample(generate, max_attempts=10)
    assert calls == list(range(1, 11))
    assert result["raw_response"] == "try-10"
    assert result["reject_resample"]["accepted"] is False
    assert result["reject_resample"]["attempts_used"] == 10
    assert result["reject_resample"]["rejected_count"] == 10


def test_rejects_empty_or_oversized_attempt_lists() -> None:
    try:
        package_reject_resample_result([], max_attempts=10)
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("empty attempts must fail")

    try:
        package_reject_resample_result([_attempt(True)] * 3, max_attempts=2)
    except ValueError as error:
        assert "max_attempts" in str(error)
    else:
        raise AssertionError("oversized attempts must fail")
