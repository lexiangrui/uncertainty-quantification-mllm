from __future__ import annotations

from pathlib import Path

from .base import GeneratedResponse, GenerationBackend, GenerationRequest


class VLLMMultimodalBackend(GenerationBackend):
    """vLLM offline engine with per-request decoding parameters.

    Passing a request window larger than ``max_num_seqs`` lets vLLM refill
    finished slots continuously while keeping the physical batch bounded.
    """

    def __init__(
        self,
        family: str,
        model_path: Path,
        *,
        adapter_path: Path | None,
        max_num_seqs: int,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ) -> None:
        if not model_path.is_dir():
            raise NotADirectoryError(model_path)
        if adapter_path is not None and not adapter_path.is_dir():
            raise NotADirectoryError(adapter_path)
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        try:
            from transformers import AutoProcessor
            from vllm import LLM
            from vllm.lora.request import LoRARequest
        except ImportError as error:
            raise RuntimeError(
                "vLLM generation requires the dedicated vLLM environment"
            ) from error

        self.family = family
        self.model_id = model_path.name
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        dtype = "float16" if family == "llava_1_5" else "bfloat16"
        self.engine = LLM(
            model=str(model_path),
            dtype=dtype,
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=adapter_path is not None,
            max_lora_rank=8,
            limit_mm_per_prompt={"image": 1},
            disable_log_stats=True,
            generation_config="vllm",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.lora_request = (
            LoRARequest("format-adapter", 1, str(adapter_path))
            if adapter_path is not None
            else None
        )

    @property
    def runtime_config(self) -> dict:
        try:
            import vllm

            version = vllm.__version__
        except (ImportError, AttributeError):
            version = "unknown"
        return {
            "engine": "vllm",
            "engine_version": version,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "local_files_only": True,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
        }

    def decode_generated_tokens(self, token_ids: tuple[int, ...]) -> str:
        return self.processor.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _render(self, request: GenerationRequest) -> dict:
        content: list[dict] = []
        if request.image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": request.prompt.user})
        messages = [{"role": "user", "content": content}]
        if request.prompt.system:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": request.prompt.system}],
                },
            )
        rendered = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        value: dict = {"prompt": rendered}
        if request.image is not None:
            value["multi_modal_data"] = {"image": request.image}
        return value

    @staticmethod
    def _chosen_log_probs(completion) -> tuple[float, ...]:
        if completion.logprobs is None:
            raise RuntimeError("vLLM did not return completion log probabilities")
        values: list[float] = []
        for token_id, candidates in zip(
            completion.token_ids, completion.logprobs, strict=True
        ):
            selected = candidates.get(token_id)
            if selected is None:
                raise RuntimeError(
                    f"vLLM logprobs omitted selected token {token_id}"
                )
            values.append(float(selected.logprob))
        return tuple(values)

    def generate_requests(
        self, requests: list[GenerationRequest], *, max_new_tokens: int
    ) -> dict[str, GeneratedResponse]:
        if not requests:
            return {}
        from vllm import SamplingParams

        prompts = [self._render(request) for request in requests]
        params = [
            SamplingParams(
                n=1,
                temperature=0.0 if request.role == "greedy" else 1.0,
                max_tokens=max_new_tokens,
                seed=request.seed,
                logprobs=1,
                stop=["</answer>"],
                include_stop_str_in_output=True,
            )
            for request in requests
        ]
        outputs = self.engine.generate(
            prompts,
            params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        if len(outputs) != len(requests):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} requests; expected {len(requests)}"
            )
        generated: dict[str, GeneratedResponse] = {}
        for request, output in zip(requests, outputs, strict=True):
            if len(output.outputs) != 1:
                raise RuntimeError(
                    f"vLLM returned {len(output.outputs)} candidates for {request.request_id}"
                )
            completion = output.outputs[0]
            token_ids = tuple(int(value) for value in completion.token_ids)
            token_log_probs = self._chosen_log_probs(completion)
            text = self.decode_generated_tokens(token_ids)
            generated[request.request_id] = GeneratedResponse(
                text=text,
                token_ids=token_ids,
                token_log_probs=token_log_probs,
                sampling_token_log_probs=token_log_probs,
                finish_reason=completion.finish_reason,
            )
        return generated
