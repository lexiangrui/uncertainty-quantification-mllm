from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from src.utils.prompts import load_prompt

from ._json import parse_json_object


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_JUDGE_PROMPT = load_prompt(
    _PROJECT_ROOT / "prompts" / "judge" / "closed_source_judge.md"
)
JUDGE_PROMPT_SHA256 = _JUDGE_PROMPT.sha256
JUDGE_SYSTEM_PROMPT = _JUDGE_PROMPT.text


@dataclass(frozen=True)
class JudgeResult:
    analysis: str
    correct: bool
    rating: int
    hallucination: bool
    hallucination_types: tuple[str, ...]
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "correct": self.correct,
            "rating": self.rating,
            "hallucination": self.hallucination,
            "hallucination_types": list(self.hallucination_types),
            "raw_response": self.raw_response,
        }


_JUDGE_MAX_IMAGE_EDGE = 1024


def _image_data_url(image: Image.Image) -> str:
    """Encode an image as a data URL, downscaled to fit within a bounding box.

    Large images cause upstream timeouts on some OpenAI-compatible proxies;
    resizing to a bounded edge keeps the payload and visual-token count
    manageable while preserving enough detail for hallucination judging.
    """
    resized = image
    if max(image.size) > _JUDGE_MAX_IMAGE_EDGE:
        ratio = _JUDGE_MAX_IMAGE_EDGE / max(image.size)
        resized = image.resize(
            (max(1, int(image.size[0] * ratio)), max(1, int(image.size[1] * ratio))),
            Image.LANCZOS,
        )
    if resized.mode in ("RGBA", "LA", "P"):
        resized = resized.convert("RGB")
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=85)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_closed_source_messages(
    *,
    dataset: str,
    question: str,
    references: list[str],
    vision: str,
    reasoning: str,
    answer: str,
    image: Image.Image | None,
) -> list[dict[str, Any]]:
    payload = {
        "dataset": dataset,
        "question": question,
        "accepted_reference_answers": references,
        "candidate_response": {
            "visual_observations": vision,
            "reasoning": reasoning,
            "answer": answer,
        },
    }
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(image), "detail": "high"},
            }
        )
    content.append(
        {
            "type": "text",
            "text": (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n\nReturn the requested json object only."
            ),
        }
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _build_responses_input(
    *,
    dataset: str,
    question: str,
    references: list[str],
    vision: str,
    reasoning: str,
    answer: str,
    image: Image.Image | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Build (system_prompt, user_content_blocks) for the OpenAI Responses API."""
    payload = {
        "dataset": dataset,
        "question": question,
        "accepted_reference_answers": references,
        "candidate_response": {
            "visual_observations": vision,
            "reasoning": reasoning,
            "answer": answer,
        },
    }
    blocks: list[dict[str, Any]] = []
    if image is not None:
        blocks.append(
            {"type": "input_image", "image_url": _image_data_url(image)}
        )
    blocks.append(
        {
            "type": "input_text",
            "text": (
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                + "\n\nReturn the requested json object only."
            ),
        }
    )
    return JUDGE_SYSTEM_PROMPT, blocks


class JudgeResponseError(ValueError):
    """Invalid judge output, retaining the raw response for audit logging."""

    def __init__(self, message: str, raw_response: str) -> None:
        super().__init__(message)
        self.raw_response = raw_response


def parse_closed_source_response(text: str) -> JudgeResult:
    try:
        value = parse_json_object(text)
    except json.JSONDecodeError as error:
        raise JudgeResponseError("judge response is not valid JSON", text) from error
    required = {"analysis", "correct", "rating", "hallucination_types"}
    if not isinstance(value, dict) or set(value) != required:
        raise JudgeResponseError("judge response has invalid fields", text)
    if not isinstance(value["analysis"], str) or not value["analysis"].strip():
        raise JudgeResponseError("analysis must be a non-empty string", text)
    if type(value["correct"]) is not bool:
        raise JudgeResponseError("correct must be a boolean", text)
    if type(value["rating"]) is not int or not 0 <= value["rating"] <= 6:
        raise JudgeResponseError("rating must be an integer from 0 through 6", text)
    types = value["hallucination_types"]
    allowed = {"vision_hallucination", "reasoning_hallucination"}
    if (
        not isinstance(types, list)
        or any(not isinstance(item, str) or item not in allowed for item in types)
        or len(types) != len(set(types))
    ):
        raise JudgeResponseError("hallucination_types is invalid", text)
    hallucination = value["rating"] < 3
    if hallucination != bool(types):
        raise JudgeResponseError("rating and hallucination_types are inconsistent", text)
    return JudgeResult(
        analysis=value["analysis"].strip(),
        correct=value["correct"],
        rating=value["rating"],
        hallucination=hallucination,
        hallucination_types=tuple(types),
        raw_response=text,
    )


def _load_local_credentials() -> tuple[str, str]:
    """Resolve base_url and api_key from the project-local .ven secrets file."""
    env_path = _PROJECT_ROOT / ".ven"
    values: dict[str, str] = {}
    if env_path.is_file():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in ("OPENAI_BASE_URL", "OPENAI_API_KEY") and key not in values:
                    values[key] = value
        except OSError:
            pass
    return values.get("OPENAI_BASE_URL", ""), values.get("OPENAI_API_KEY", "")


class ClosedSourceJudge:
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 512,
        timeout_seconds: float = 90.0,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if client is None:
            base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not base_url or not api_key:
                fallback_url, fallback_key = _load_local_credentials()
                base_url = base_url or fallback_url
                api_key = api_key or fallback_key
            if not base_url or not api_key:
                raise RuntimeError(
                    "OPENAI_BASE_URL and OPENAI_API_KEY must be set as environment "
                    "variables or listed in the project-local .ven file"
                )
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_seconds)
        self.model = model
        self.max_tokens = max_tokens
        self.client = client

    def judge(self, **message_inputs: Any) -> JudgeResult:
        system_prompt, user_content = _build_responses_input(**message_inputs)
        response = self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            text={"format": {"type": "json_object"}},
            max_output_tokens=self.max_tokens,
        )
        content = response.output_text
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("judge returned empty content")
        return parse_closed_source_response(content)
