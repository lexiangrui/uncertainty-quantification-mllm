import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "judge_hallucination_mmhal.py"
SPEC = importlib.util.spec_from_file_location("judge_hallucination_mmhal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("rating, hallucination", [(0, True), (2, True), (3, False), (6, False)])
def test_parse_response_uses_official_threshold(rating, hallucination):
    parsed = MODULE.parse_response(
        f'{{"analysis":"brief","correct":true,"rating":{rating}}}'
    )
    assert parsed["hallucination"] is hallucination


@pytest.mark.parametrize(
    "text",
    [
        '{"analysis":"brief","correct":true,"rating":7}',
        '{"analysis":"brief","correct":"true","rating":4}',
        '{"correct":true,"rating":4}',
        '```json {"analysis":"brief","correct":true,"rating":4} ```',
    ],
)
def test_parse_response_rejects_invalid_schema(text):
    with pytest.raises(ValueError):
        MODULE.parse_response(text)
