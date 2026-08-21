"""vLLM generation backend used only for the hybrid generation stage.

This module deliberately owns no hidden-state or attention extraction.  The
HF replay stage consumes the exact token IDs emitted here for all internal
signals required by UQ and ERA.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.base import GeneratedResponse, GenerationBackend, GenerationRequest
from src.models.internvl import INTERNVL_SYSTEM_PROMPT


class VLLMMultimodalBackend(GenerationBackend):
    def __init__(
        self,
        family: str,
        model_path: Path,
        *,
        adapter_path: Path | None,
        max_num_seqs: int = 8,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
    ) -> None:
        if not model_path.is_dir():
            raise NotADirectoryError(model_path)
        if family not in {"internvl3_5", "internvl3_5_original", "qwen2_5_vl", "llava_1_5"}:
            raise ValueError(f"unknown model family: {family}")
        if adapter_path is not None and not adapter_path.is_dir():
            raise NotADirectoryError(adapter_path)
        self.family = family
        self.model_path = model_path
        self.model_id = model_path.name
        self.adapter_path = adapter_path
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.processor: Any | None = None
        self.tokenizer: Any | None = None
        self.llm: Any | None = None
        self.lora_request: Any | None = None

    @property
    def runtime_config(self) -> dict:
        return {
            "engine": "vllm",
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "local_files_only": True,
            "multimodal_inputs": "offline_pil",
        }

    def _load(self) -> None:
        if self.llm is not None:
            return
        try:
            from transformers import AutoProcessor, AutoTokenizer
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise RuntimeError("vLLM generation requires transformers in the vLLM environment") from error
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=True
            )
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        except Exception:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False,
            )
        kwargs = {
            "model": str(self.model_path),
            "trust_remote_code": True,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "limit_mm_per_prompt": {"image": 1},
        }
        if self.adapter_path is not None:
            kwargs["enable_lora"] = True
            kwargs["max_loras"] = 1
        self.llm = LLM(**kwargs)
        if self.adapter_path is not None:
            self.lora_request = LoRARequest("experiment-adapter", 1, str(self.adapter_path))

    def decode_generated_tokens(self, token_ids: tuple[int, ...]) -> str:
        self._load()
        assert self.tokenizer is not None
        return self.tokenizer.decode(
            list(token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

    def _render(self, request: GenerationRequest) -> dict:
        self._load()
        renderer = self.processor or self.tokenizer
        assert renderer is not None
        content: list[dict] = []
        if request.image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": request.prompt.user})
        messages = [{"role": "user", "content": content}]
        system = request.prompt.system
        if self.family == "internvl3_5_original" and not system:
            system = INTERNVL_SYSTEM_PROMPT
        if system:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system}],
                },
            )
        if not hasattr(renderer, "apply_chat_template"):
            raise RuntimeError(f"{self.family} tokenizer has no chat template")
        prompt = renderer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        value = {"prompt": prompt}
        if request.image is not None:
            value["multi_modal_data"] = {"image": request.image}
        return value

    @staticmethod
    def _logprob_value(entry: Any, token_id: int) -> float:
        if entry is None:
            raise RuntimeError("vLLM omitted output log probabilities")
        value = entry.get(token_id) if hasattr(entry, "get") else None
        if value is None and hasattr(entry, "get"):
            value = entry.get(str(token_id))
        if value is None:
            raise RuntimeError(f"vLLM omitted chosen token log probability for token {token_id}")
        result = getattr(value, "logprob", None)
        if result is None and isinstance(value, dict):
            result = value.get("logprob")
        if result is None:
            raise RuntimeError(f"unrecognized vLLM logprob entry for token {token_id}")
        return float(result)

    def generate_requests(
        self, requests: list[GenerationRequest], *, max_new_tokens: int
    ) -> dict[str, GeneratedResponse]:
        if not requests:
            return {}
        self._load()
        assert self.llm is not None
        from vllm import SamplingParams

        roles = {request.role for request in requests}
        if len(roles) != 1:
            raise ValueError("vLLM batches must contain one decoding role")
        do_sample = requests[0].role == "sample"
        params = [
            SamplingParams(
                temperature=1.0 if do_sample else 0.0,
                max_tokens=max_new_tokens,
                n=1,
                logprobs=1,
                stop=["</answer>"],
                include_stop_str_in_output=True,
                seed=request.seed,
            )
            for request in requests
        ]
        inputs = [self._render(request) for request in requests]
        outputs = self.llm.generate(
            inputs,
            params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        if len(outputs) != len(requests):
            raise RuntimeError(f"vLLM returned {len(outputs)} requests; expected {len(requests)}")
        values: dict[str, GeneratedResponse] = {}
        for request, result in zip(requests, outputs, strict=True):
            if not result.outputs:
                raise RuntimeError(f"vLLM returned no completion for {request.request_id}")
            completion = result.outputs[0]
            token_ids = tuple(int(value) for value in completion.token_ids)
            token_log_probs = tuple(
                self._logprob_value(entry, token_id)
                for entry, token_id in zip(completion.logprobs or (), token_ids, strict=True)
            )
            values[request.request_id] = GeneratedResponse(
                text=str(completion.text).strip(),
                token_ids=token_ids,
                token_log_probs=token_log_probs,
                sampling_token_log_probs=token_log_probs,
                finish_reason=completion.finish_reason,
                rng_seed=request.seed,
            )
        return values
