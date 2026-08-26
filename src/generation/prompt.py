from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.prompts import load_prompt


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_XML_PROMPT = load_prompt(
    _PROJECT_ROOT / "prompts" / "LoRA" / "xml_lora_instruction.md"
)
XML_LORA_PROMPT_SHA256 = _XML_PROMPT.sha256
_NATIVE_THREE_PART_PROMPT = load_prompt(
    _PROJECT_ROOT / "prompts" / "generation" / "native_three_part_zero_shot.md"
)
NATIVE_THREE_PART_PROMPT_SHA256 = _NATIVE_THREE_PART_PROMPT.sha256


@dataclass(frozen=True)
class GenerationPrompt:
    system: str
    user: str
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptSpec:
    response_format: str
    prompt_sha256: str


PROMPT_SPECS = {
    "xml_lora": PromptSpec("xml", XML_LORA_PROMPT_SHA256),
    "native_three_part": PromptSpec(
        "plain_sections", NATIVE_THREE_PART_PROMPT_SHA256
    ),
}


def get_prompt_spec(style: str) -> PromptSpec:
    try:
        return PROMPT_SPECS[style]
    except KeyError as error:
        raise ValueError(f"unknown prompt style: {style}") from error


def build_prompt(
    question: str, has_image: bool, style: str = "xml_lora"
) -> GenerationPrompt:
    get_prompt_spec(style)
    image_line = "[Image]\nThe image is attached to this message.\n\n" if has_image else ""
    prompt = _XML_PROMPT if style == "xml_lora" else _NATIVE_THREE_PART_PROMPT
    user = f"{prompt.text}\n\n{image_line}[Question]\n{question.strip()}"
    stop = ("</answer>",) if style == "xml_lora" else ()
    return GenerationPrompt(system="", user=user, stop=stop)
