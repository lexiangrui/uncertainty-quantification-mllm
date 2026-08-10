"""Build and train grounded XML supervision for multimodal models."""

from .validation import ValidationError, validate_teacher_payload
from .xml import build_xml_response
from .prompts import LORA_XML_INSTRUCTION, LORA_XML_PROMPT_SHA256, LORA_XML_PROMPT_VERSION

__all__ = [
    "ValidationError",
    "build_xml_response",
    "validate_teacher_payload",
    "LORA_XML_INSTRUCTION",
    "LORA_XML_PROMPT_VERSION",
    "LORA_XML_PROMPT_SHA256",
]
