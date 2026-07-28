from __future__ import annotations

from xml.sax.saxutils import escape


def build_xml_response(vision: str, reasoning: str, answer: str) -> str:
    """Wrap already validated text; the teacher never controls the XML structure."""
    return (
        f"<vision>{escape(vision)}</vision>"
        f"<reasoning>{escape(reasoning)}</reasoning>"
        f"<answer>{escape(answer)}</answer>"
    )
