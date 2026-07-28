import pytest

from lora_format.llava_sft import append_eos


def test_append_eos_once() -> None:
    assert append_eos("<answer>yes</answer>", "</s>") == "<answer>yes</answer></s>"
    assert append_eos("<answer>yes</answer></s>", "</s>") == "<answer>yes</answer></s>"


def test_missing_eos_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must define an EOS"):
        append_eos("answer", None)
