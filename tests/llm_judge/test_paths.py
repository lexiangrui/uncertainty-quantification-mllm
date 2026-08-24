from pathlib import Path

import pytest

from src.llm_judge.paths import judge_directory_name, validate_raw_judge_output


def test_judge_directory_name_is_model_specific() -> None:
    assert judge_directory_name("gpt-5.6-terra") == "judging_gpt_5_6_terra"
    assert judge_directory_name("gemini-3.7-flash") == "judging_gemini_3_7_flash"


def test_raw_output_cannot_use_official_judging_directory() -> None:
    with pytest.raises(ValueError, match="judging_gpt_5_6_terra"):
        validate_raw_judge_output(Path("results/judging/llava/vilp.jsonl"), "gpt-5.6-terra")
    validate_raw_judge_output(
        Path("results/judging_gpt_5_6_terra/llava/vilp.jsonl"),
        "gpt-5.6-terra",
    )
