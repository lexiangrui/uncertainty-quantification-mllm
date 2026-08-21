from PIL import Image

from scripts.generation.replay_hf_artifacts import _run_replay_calls
from src.generation.prompt import GenerationPrompt
from src.models.base import GeneratedResponse, GenerationRequest


def _request(name: str, *, image=None) -> GenerationRequest:
    return GenerationRequest(
        request_id=name,
        sample_id=name,
        role="greedy",
        draw_index=None,
        seed=1,
        image=image,
        prompt=GenerationPrompt(system="", user="Question"),
    )


class Backend:
    def __init__(self):
        self.modalities = []

    def teacher_force_responses(self, requests, token_sequences):
        modalities = {request.image is not None for request in requests}
        assert len(modalities) == 1
        self.modalities.append((next(iter(modalities)), len(requests)))
        return {
            request.request_id: GeneratedResponse(
                text="answer",
                token_ids=token_ids,
                token_log_probs=(-0.1,) * len(token_ids),
                sampling_token_log_probs=(-0.1,) * len(token_ids),
            )
            for request, token_ids in zip(requests, token_sequences, strict=True)
        }


def test_replay_calls_split_hallusionbench_modality_boundary() -> None:
    backend = Backend()
    calls = [
        *[(_request(f"image-{index}", image=Image.new("RGB", (2, 2))), (1,)) for index in range(1)],
        *[(_request(f"text-{index}"), (1,)) for index in range(4)],
    ]

    responses = _run_replay_calls(backend, calls, batch_size=5)

    assert len(responses) == 5
    assert backend.modalities == [(True, 1), (False, 4)]
