from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image


JUDGE_PROMPT_VERSION = "correctness-hallucination-v1"

JUDGE_SYSTEM_PROMPT = """You are an impartial judge of a multimodal model response.

Evaluate two independent properties:

1. Correctness: Judge only the Answer part against the accepted reference
answers. Do not use the Visual observations or Reasoning parts to repair the
answer.

2. Hallucination: Judge only the Visual observations and Reasoning parts.
Hallucination means a factual claim that is unsupported by or inconsistent with
the image, question, or accepted reference information. A wrong calculation or
invalid deduction is not automatically a hallucination unless it introduces an
unsupported factual premise.

Assign one MMHal-style rating:
- 6: very informative with good analysis or reasoning, no hallucination
- 5: very informative, no hallucination
- 4: somewhat informative, no hallucination
- 3: not informative, no hallucination
- 2: very informative, with hallucination
- 1: somewhat informative, with hallucination
- 0: not informative, with hallucination

If the rating is 0, 1, or 2, hallucination_types must contain
"vision_hallucination", "reasoning_hallucination", or both. If the rating is
3, 4, 5, or 6, hallucination_types must be an empty array.

Return one json object with exactly these fields:
{"analysis":"brief justification","correct":true,"rating":4,"hallucination_types":[]}"""


@dataclass(frozen=True)
class JudgeResult:
    analysis: str
    correct: bool
    rating: int
    hallucination: bool
    hallucination_types: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis,
            "correct": self.correct,
            "rating": self.rating,
            "hallucination": self.hallucination,
            "hallucination_types": list(self.hallucination_types),
        }


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def build_openai_judge_messages(
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


def parse_openai_judge_response(text: str) -> JudgeResult:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as error:
        raise ValueError("judge response is not valid JSON") from error
    required = {"analysis", "correct", "rating", "hallucination_types"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("judge response has invalid fields")
    if not isinstance(value["analysis"], str) or not value["analysis"].strip():
        raise ValueError("analysis must be a non-empty string")
    if type(value["correct"]) is not bool:
        raise ValueError("correct must be a boolean")
    if type(value["rating"]) is not int or not 0 <= value["rating"] <= 6:
        raise ValueError("rating must be an integer from 0 through 6")
    types = value["hallucination_types"]
    allowed = {"vision_hallucination", "reasoning_hallucination"}
    if (
        not isinstance(types, list)
        or any(not isinstance(item, str) or item not in allowed for item in types)
        or len(types) != len(set(types))
    ):
        raise ValueError("hallucination_types is invalid")
    hallucination = value["rating"] < 3
    if hallucination != bool(types):
        raise ValueError("rating and hallucination_types are inconsistent")
    return JudgeResult(
        analysis=value["analysis"].strip(),
        correct=value["correct"],
        rating=value["rating"],
        hallucination=hallucination,
        hallucination_types=tuple(types),
    )


class OpenAIChatJudge:
    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 512,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if client is None:
            base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not base_url or not api_key:
                raise RuntimeError(
                    "OPENAI_BASE_URL and OPENAI_API_KEY must be non-empty environment variables"
                )
            from openai import OpenAI

            client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.client = client
        self.last_raw_response: str | None = None

    def judge(self, **message_inputs: Any) -> JudgeResult:
        self.last_raw_response = None
        messages = build_openai_judge_messages(**message_inputs)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        if len(completion.choices) != 1:
            raise RuntimeError("judge returned an unexpected number of choices")
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("judge returned empty content")
        self.last_raw_response = content
        return parse_openai_judge_response(content)
