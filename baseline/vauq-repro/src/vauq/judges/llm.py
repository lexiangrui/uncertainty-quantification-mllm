"""LLM-as-judge for free-form VQA answers.

The default configuration targets the Krill OpenAI-compatible endpoint with
``deepseek-v4-flash:free`` through the OpenAI Python SDK.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import OpenAI

from .base import Judge


class LLMJudge(Judge):
    """Semantic correctness judge backed by a chat-completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        _load_default_env_file()
        self.api_key = (
            api_key
            or os.environ.get("LLM_JUDGE_API_KEY")
            or os.environ.get("KRILL_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "LLM_JUDGE_API_KEY is required for --judge llm. "
                "Export it in the shell or pass labels with another judge."
            )
        self.base_url = (
            base_url
            or os.environ.get("LLM_JUDGE_BASE_URL")
            or os.environ.get("KRILL_BASE_URL")
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.cdn-krill-ai.com/coding/v1"
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("LLM_JUDGE_MODEL")
            or os.environ.get("KRILL_JUDGE_MODEL")
            or os.environ.get("DEEPSEEK_JUDGE_MODEL")
            or os.environ.get("DEEPSEEK_MODEL")
            or "deepseek-v4-flash:free"
        )
        self.timeout = timeout or float(os.environ.get("LLM_JUDGE_TIMEOUT", os.environ.get("DEEPSEEK_TIMEOUT", "60")))
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("LLM_JUDGE_MAX_RETRIES", os.environ.get("DEEPSEEK_MAX_RETRIES", "3")))
        self.last_result: dict[str, Any] | None = None

    def judge(self, prediction: str, gold: Any, sample: dict[str, Any] | None = None) -> bool | None:
        prompt = _build_prompt(prediction=prediction, gold=gold, sample=sample)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 8,
            "stream": False,
        }
        content = self._post_chat(payload)
        correct = _parse_verdict(content)
        self.last_result = {
            "correct": correct,
            "raw_response": content.strip()[:1000],
            "model": self.model,
            "base_url": self.base_url,
        }
        return correct

    def _post_chat(self, payload: dict[str, Any]) -> str:
        """Call the chat endpoint with defensive retry for empty/malformed replies.

        The OpenAI SDK's built-in ``max_retries`` only retries on connection errors
        and certain HTTP status codes (429/5xx). Free-tier endpoints sometimes
        return HTTP 200 with ``completion=None`` or empty ``choices`` (rate limiting,
        upstream timeouts), which the SDK does not retry and which previously crashed
        the whole judge run mid-loop. We add an inner backoff loop that treats those
        malformed 200s as retryable.
        """
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        inner_retries = max(1, int(os.environ.get("LLM_JUDGE_INNER_RETRIES", "4")))
        backoff = 2.0
        last_err: Exception | None = None
        for _ in range(inner_retries):
            try:
                completion = client.chat.completions.create(**payload)
            except Exception as exc:  # noqa: BLE001 - surface after exhausting retries
                last_err = exc
            else:
                choices = getattr(completion, "choices", None)
                if not choices:
                    last_err = RuntimeError("LLM judge returned empty completion (no choices)")
                else:
                    content = getattr(choices[0].message, "content", None)
                    if content:
                        return content
                    last_err = RuntimeError("LLM judge returned empty content")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
        raise RuntimeError(f"LLM judge failed after {inner_retries} attempts: {last_err}")


_SYSTEM_PROMPT = """You are a strict but fair judge for visual question answering.
Reply with exactly one word: CORRECT or WRONG.

Mark CORRECT when the model answer semantically matches the ground truth answer,
even if wording differs. Mark WRONG if it contradicts the ground truth, gives a
different entity/number/color/action, refuses, or is too vague to verify."""


def _build_prompt(prediction: str, gold: Any, sample: dict[str, Any] | None) -> str:
    sample = sample or {}
    question = sample.get("question") or ""
    choices = sample.get("choices")
    subset = sample.get("subset")
    return json.dumps(
        {
            "task": "Judge whether the model answer is correct. Reply only CORRECT or WRONG.",
            "question": question,
            "ground_truth_answer": gold,
            "model_answer": prediction,
            "choices": choices,
            "subset": subset,
        },
        ensure_ascii=False,
    )


def _parse_verdict(content: str) -> bool:
    text = content.strip()
    if not text:
        raise ValueError("LLM judge returned empty content")
    normalized = text.upper()
    match = re.search(r"\b(INCORRECT|WRONG|FALSE|CORRECT|TRUE)\b", normalized)
    if not match:
        raise ValueError(f"LLM judge returned unparsable verdict: {text[:200]!r}")
    verdict = match.group(1)
    return verdict in {"CORRECT", "TRUE"}


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if not text:
        raise ValueError("LLM judge returned empty content")
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1], strict=False)
        raise


def _load_default_env_file() -> None:
    env_path = _repo_root() / "configs" / "llm_judge.env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[3]
