import base64
import json

import pytest

from lora_format.teacher import build_messages, request_teacher_payload, require_api_config


def test_blank_api_configuration_fails_before_request(monkeypatch) -> None:
    monkeypatch.setenv("QWEN_TEACHER_BASE_URL", "")
    monkeypatch.setenv("QWEN_TEACHER_API_KEY", "")
    with pytest.raises(RuntimeError, match="must both be non-empty"):
        require_api_config()


def test_messages_contain_image_question_and_answer(tmp_path) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"jpeg")
    row = {"question": "What color?", "answer": "blue"}
    messages = build_messages(row, image, "system", [])
    assert messages[0] == {"role": "system", "content": "system"}
    content = messages[1]["content"]
    assert "What color?" in content[0]["text"]
    assert "blue" in content[0]["text"]
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == b"jpeg"


class _Completions:
    def create(self, **kwargs):
        assert kwargs["model"] == "qwen3.7-plus"
        assert kwargs["temperature"] == 0
        assert kwargs["response_format"] == {"type": "json_object"}
        message = type("Message", (), {"content": json.dumps({"vision": "v", "reasoning": "r", "answer": "a"})})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})


class _Client:
    chat = type("Chat", (), {"completions": _Completions()})


def test_chat_completions_protocol_is_used() -> None:
    assert request_teacher_payload(_Client(), "qwen3.7-plus", []) == {
        "vision": "v",
        "reasoning": "r",
        "answer": "a",
    }


class _FencedCompletions:
    def create(self, **kwargs):
        content = '```json\n{"vision":"v","reasoning":"r","answer":"a"}\n```'
        message = type("Message", (), {"content": content})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})


class _FencedClient:
    chat = type("Chat", (), {"completions": _FencedCompletions()})


def test_json_markdown_fence_is_unwrapped() -> None:
    assert request_teacher_payload(_FencedClient(), "qwen3.7-plus", []) == {
        "vision": "v",
        "reasoning": "r",
        "answer": "a",
    }


class _CommentaryCompletions:
    def create(self, **kwargs):
        message = type("Message", (), {"content": 'Here is the result:\n```json\n{}\n```'})
        choice = type("Choice", (), {"message": message})
        return type("Response", (), {"choices": [choice]})


class _CommentaryClient:
    chat = type("Chat", (), {"completions": _CommentaryCompletions()})


def test_json_fence_with_extra_commentary_is_rejected() -> None:
    with pytest.raises(json.JSONDecodeError):
        request_teacher_payload(_CommentaryClient(), "qwen3.7-plus", [])
