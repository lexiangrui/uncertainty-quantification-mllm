from PIL import Image

from src.generation.prompt import GenerationPrompt
from src.models.base import GeneratedResponse, GenerationRequest
from src.models.huggingface import HuggingFaceMultimodalBackend


def request(request_id: str, *, image: bool) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        sample_id=request_id,
        role="sample",
        draw_index=0,
        seed=1,
        image=Image.new("RGB", (2, 2)) if image else None,
        prompt=GenerationPrompt(system="", user="Question"),
    )


def response() -> GeneratedResponse:
    return GeneratedResponse(
        text="answer",
        token_ids=(1,),
        token_log_probs=(-0.1,),
        sampling_token_log_probs=(-0.1,),
    )


def test_generate_requests_splits_image_and_text_modalities() -> None:
    backend = object.__new__(HuggingFaceMultimodalBackend)
    calls: list[list[GenerationRequest]] = []

    def generate_batch(requests, *, max_new_tokens):
        assert max_new_tokens == 32
        assert len({item.image is not None for item in requests}) == 1
        calls.append(requests)
        return {item.request_id: response() for item in requests}

    backend._generate_batch = generate_batch
    generated = backend.generate_requests(
        [request("image-1", image=True), request("text-1", image=False)],
        max_new_tokens=32,
    )

    assert set(generated) == {"image-1", "text-1"}
    assert len(calls) == 2
