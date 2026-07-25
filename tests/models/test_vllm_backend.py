from types import SimpleNamespace

from PIL import Image

from src.generation.prompt import GenerationPrompt
from src.models.base import GenerationRequest
from src.models.vllm_backend import VLLMMultimodalBackend


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"add_generation_prompt": True, "tokenize": False}
        return "rendered"


def backend() -> VLLMMultimodalBackend:
    value = object.__new__(VLLMMultimodalBackend)
    value.processor = FakeProcessor()
    return value


def request(image=None) -> GenerationRequest:
    return GenerationRequest(
        request_id="one:greedy:attempt-1",
        sample_id="one",
        role="greedy",
        draw_index=None,
        seed=1,
        image=image,
        prompt=GenerationPrompt(system="", user="Question"),
    )


def test_render_adds_multimodal_data_only_for_image() -> None:
    assert backend()._render(request()) == {"prompt": "rendered"}
    rendered = backend()._render(request(Image.new("RGB", (2, 2))))
    assert rendered["prompt"] == "rendered"
    assert rendered["multi_modal_data"]["image"].size == (2, 2)


def test_chosen_log_probs_follow_generated_token_ids() -> None:
    completion = SimpleNamespace(
        token_ids=[4, 8],
        logprobs=[
            {4: SimpleNamespace(logprob=-0.1), 2: SimpleNamespace(logprob=-1.0)},
            {8: SimpleNamespace(logprob=-0.2)},
        ],
    )
    assert VLLMMultimodalBackend._chosen_log_probs(completion) == (-0.1, -0.2)
