from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path


PROMPT_VERSION = "qwen-vqav2-grounded-v2"
JSON_FENCE = re.compile(r"\A\s*```json\s*\n(?P<payload>.*?)\n```\s*\Z", re.DOTALL | re.IGNORECASE)


def require_api_config() -> tuple[str, str]:
    base_url = os.environ.get("QWEN_TEACHER_BASE_URL", "").strip()
    api_key = os.environ.get("QWEN_TEACHER_API_KEY", "").strip()
    if not base_url or not api_key:
        raise RuntimeError(
            "QWEN_TEACHER_BASE_URL and QWEN_TEACHER_API_KEY must both be non-empty"
        )
    return base_url, api_key


def image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_messages(
    row: dict,
    image_path: Path,
    system_prompt: str,
    examples: list[dict],
) -> list[dict]:
    examples_text = json.dumps(examples, ensure_ascii=False, indent=2)
    user_text = (
        "The following examples show the desired response structure and image-grounded writing. "
        "Each uses the image description provided within that example:\n"
        f"{examples_text}\n\n"
        "Inspect the attached image and create a response for this question.\n"
        f"Question: {row['question']}\n"
        f"Use this exact answer in the `answer` field: {row['answer']}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            ],
        },
    ]


def create_teacher_client():
    from openai import OpenAI

    base_url, api_key = require_api_config()
    return OpenAI(base_url=base_url, api_key=api_key, timeout=180.0, max_retries=3)


def request_teacher_payload(client, model: str, messages: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("teacher returned empty content")
    match = JSON_FENCE.fullmatch(content)
    if match:
        content = match.group("payload")
    return json.loads(content)
