import hashlib
from pathlib import Path

import pytest

from lora_format.llava_sft import append_eos
from lora_format.prompts import (
    LORA_XML_INSTRUCTION,
    LORA_XML_PROMPT_SHA256,
    LORA_XML_PROMPT_VERSION,
)


def test_append_eos_once() -> None:
    assert append_eos("<answer>yes</answer>", "</s>") == "<answer>yes</answer></s>"
    assert append_eos("<answer>yes</answer></s>", "</s>") == "<answer>yes</answer></s>"


def test_missing_eos_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="must define an EOS"):
        append_eos("answer", None)


def test_lora_prompt_loads_from_versioned_repository_file() -> None:
    path = Path(__file__).resolve().parents[2] / "prompts" / "LoRA" / "xml_lora_instruction_v1.md"
    text = path.read_text(encoding="utf-8").strip()
    assert LORA_XML_PROMPT_VERSION == "xml-lora-instruction-v1"
    assert LORA_XML_INSTRUCTION == text
    assert LORA_XML_PROMPT_SHA256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
