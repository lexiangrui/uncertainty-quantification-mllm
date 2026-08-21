from pathlib import Path

from src.generation.prompt import GenerationPrompt
from src.models.base import GenerationRequest
from src.models.internvl import INTERNVL_SYSTEM_PROMPT
from src.models.vllm_backend import VLLMMultimodalBackend


class Renderer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "rendered"


def test_original_internvl_vllm_uses_official_system(tmp_path: Path) -> None:
    backend = VLLMMultimodalBackend(
        "internvl3_5_original", tmp_path, adapter_path=None
    )
    renderer = Renderer()
    backend.processor = renderer
    backend.llm = object()
    request = GenerationRequest(
        request_id="r",
        sample_id="s",
        role="greedy",
        draw_index=None,
        seed=1,
        image=None,
        prompt=GenerationPrompt(system="", user="Question"),
    )
    assert backend._render(request)["prompt"] == "rendered"
    assert renderer.messages[0]["role"] == "system"
    assert renderer.messages[0]["content"][0]["text"] == INTERNVL_SYSTEM_PROMPT
