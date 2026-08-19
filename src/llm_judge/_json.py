"""Small, strict-tolerant JSON extraction for judge model responses."""

from __future__ import annotations

import json


def parse_json_object(text: str) -> object:
    """Decode one JSON object, allowing a Markdown fence or surrounding prose."""
    stripped = text.strip()
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        if newline >= 0:
            stripped = stripped[newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object found", stripped, 0)
    return json.loads(stripped[start : end + 1])
