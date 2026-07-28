"""Build and train grounded XML supervision for multimodal models."""

from .validation import ValidationError, validate_teacher_payload
from .xml import build_xml_response

__all__ = ["ValidationError", "build_xml_response", "validate_teacher_payload"]
