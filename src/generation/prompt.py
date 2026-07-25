from __future__ import annotations

from dataclasses import dataclass

from .xml_format import XML_ZERO_SHOT_INSTRUCTION


XML_LORA_PROMPT_VERSION = "xml-lora-zero-shot-v1"


@dataclass(frozen=True)
class GenerationPrompt:
    system: str
    user: str


@dataclass(frozen=True)
class PromptSpec:
    version: str
    response_format: str


PROMPT_SPECS = {
    "xml_lora": PromptSpec(XML_LORA_PROMPT_VERSION, "xml"),
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
    user = f"{XML_ZERO_SHOT_INSTRUCTION}\n\n{image_line}[Question]\n{question.strip()}"
    return GenerationPrompt(system="", user=user)
