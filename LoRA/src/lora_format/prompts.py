from __future__ import annotations

import hashlib
from pathlib import Path


LORA_XML_PROMPT_VERSION = "xml-lora-instruction-v1"
_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "LoRA" / "xml_lora_instruction_v1.md"
LORA_XML_INSTRUCTION = _PROMPT_PATH.read_text(encoding="utf-8").strip()
if not LORA_XML_INSTRUCTION:
    raise ValueError(f"prompt file is empty: {_PROMPT_PATH}")
LORA_XML_PROMPT_SHA256 = hashlib.sha256(LORA_XML_INSTRUCTION.encode("utf-8")).hexdigest()
