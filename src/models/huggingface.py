from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from PIL import Image

from src.generation.prompt import GenerationPrompt

from .base import GeneratedResponse, GenerationBackend, GenerationRequest


class HuggingFaceMultimodalBackend(GenerationBackend):
    def __init__(
        self,
        family: str,
        model_path: Path,
        *,
        attn_implementation: str | None,
        adapter_path: Path | None,
    ) -> None:
        if not model_path.is_dir():
            raise NotADirectoryError(model_path)
        if family not in {"internvl3_5", "qwen2_5_vl", "llava_1_5"}:
            raise ValueError(f"unknown model family: {family}")

        self.family = family
        self.model_id = model_path.name
        self.model_path = model_path
        self.attn_implementation = attn_implementation
        self.adapter_path = adapter_path
        # Slurm export preserves an explicitly empty value for non-Qwen jobs;
        # treat that value the same as an unset override.
        self.device_map_name = os.environ.get("QWEN_DEVICE_MAP") or None
        if self.device_map_name is not None:
            if family != "qwen2_5_vl":
                raise ValueError("QWEN_DEVICE_MAP is only supported for qwen2_5_vl")
            if self.device_map_name != "vision_language_split":
                raise ValueError(
                    "QWEN_DEVICE_MAP must be vision_language_split when set"
                )
        self.processor = None
        self.model = None
        self.device = None

    @property
    def runtime_config(self) -> dict:
        config = {
            "engine": "transformers",
            "attn_implementation": self.attn_implementation,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "local_files_only": True,
            "adaptive_oom_split": True,
            "modality_batch_split": True,
        }
        if self.device_map_name is not None:
            config["device_map"] = self.device_map_name
        return config

    def decode_generated_tokens(self, token_ids: tuple[int, ...]) -> str:
        self._load()
        assert self.processor is not None
        return self.processor.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoProcessor

        if self.family == "internvl3_5":
            from transformers import InternVLForConditionalGeneration as model_class
        elif self.family == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration as model_class
        elif self.family == "llava_1_5":
            from transformers import LlavaForConditionalGeneration as model_class
        else:
            raise ValueError(f"unknown model family: {self.family}")

        if self.attn_implementation == "flash_attention_2":
            from transformers.utils import is_flash_attn_2_available

            if not torch.cuda.is_available() or not is_flash_attn_2_available():
                raise RuntimeError(
                    "FlashAttention2 requires an initialized CUDA device and the local "
                    "flash-attn extension. The current Slurm allocation has no usable CUDA "
                    "device; resubmit on a healthy GPU node instead of falling back to a "
                    "Hub kernel or CPU attention."
                )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True
        )
        device_map = (
            {
                "model.visual": 0,
                "model.language_model": 1,
                "lm_head": 1,
            }
            if self.device_map_name == "vision_language_split"
            else "auto"
        )
        model_kwargs = {
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        model_kwargs["dtype"] = (
            torch.float16 if self.family == "llava_1_5" else torch.bfloat16
        )
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self.model = model_class.from_pretrained(
            self.model_path, **model_kwargs
        ).eval()
        if self.attn_implementation == "flash_attention_2":
            device_map = getattr(self.model, "hf_device_map", {})
            offloaded_modules = sorted(
                name
                for name, device in device_map.items()
                if str(device) in {"cpu", "disk"}
            )
            if offloaded_modules:
                preview = ", ".join(offloaded_modules[:3])
                raise RuntimeError(
                    "FlashAttention2 cannot run with CPU/disk offload. "
                    f"Accelerate offloaded {preview}; allocate a GPU with enough free "
                    "memory or use --attn-implementation sdpa explicitly."
                )
        if self.adapter_path is not None:
            if not self.adapter_path.is_dir():
                raise NotADirectoryError(self.adapter_path)
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, self.adapter_path, local_files_only=True
            ).eval()
        self.device = next(self.model.parameters()).device

    def _semantic_embedding_module(self):
        assert self.model is not None
        full_model = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        language_model = getattr(full_model, "language_model", None)
        if language_model is None:
            core = getattr(full_model, "model", None)
            language_model = getattr(core, "language_model", None)
        if language_model is None:
            raise RuntimeError("multimodal model does not expose its language model")

        # Causal-LM heads return logits, while their decoder bodies return the
        # last hidden state needed by UMPIRE. Hook the body when it is exposed
        # separately (for example LlamaForCausalLM.model).
        decoder = getattr(language_model, "model", None)
        return decoder if decoder is not None else language_model

    @torch.inference_mode()
    def generate_requests(
        self, requests: list[GenerationRequest], *, max_new_tokens: int
    ) -> dict[str, GeneratedResponse]:
        if not requests:
            return {}
        roles = {request.role for request in requests}
        if len(roles) != 1:
            raise ValueError("Transformers batches must contain one decoding role")
        modality_groups = {
            has_image: [
                request
                for request in requests
                if (request.image is not None) == has_image
            ]
            for has_image in {request.image is not None for request in requests}
        }
        if len(modality_groups) > 1:
            generated: dict[str, GeneratedResponse] = {}
            for group in modality_groups.values():
                generated.update(
                    self.generate_requests(group, max_new_tokens=max_new_tokens)
                )
            return generated
        try:
            return self._generate_batch(requests, max_new_tokens=max_new_tokens)
        except torch.OutOfMemoryError:
            if len(requests) == 1:
                raise
            torch.cuda.empty_cache()
            middle = len(requests) // 2
            return {
                **self.generate_requests(requests[:middle], max_new_tokens=max_new_tokens),
                **self.generate_requests(requests[middle:], max_new_tokens=max_new_tokens),
            }

    @staticmethod
    def _batch_seed(requests: list[GenerationRequest]) -> int:
        value = ":".join(str(request.seed) for request in requests)
        return int.from_bytes(
            hashlib.sha256(value.encode()).digest()[:8], "big"
        ) % (2**63 - 1)

    def _batch_inputs(self, requests: list[GenerationRequest]):
        self._load()
        assert self.processor is not None and self.device is not None
        rendered: list[str] = []
        images = []
        has_images = {request.image is not None for request in requests}
        if len(has_images) != 1:
            raise ValueError("a Transformers batch cannot mix image and text-only requests")
        for request in requests:
            content: list[dict] = []
            if request.image is not None:
                content.append({"type": "image"})
                images.append(request.image)
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
            rendered.append(
                self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        tokenizer.padding_side = "left"
        kwargs = {"text": rendered, "padding": True, "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs)
        return {name: value.to(self.device) for name, value in inputs.items()}

    @torch.inference_mode()
    def _generate_batch(
        self, requests: list[GenerationRequest], *, max_new_tokens: int
    ) -> dict[str, GeneratedResponse]:
        inputs = self._batch_inputs(requests)
        assert self.model is not None and self.processor is not None
        do_sample = requests[0].role == "sample"
        batch_seed = self._batch_seed(requests)
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        prompt_width = int(inputs["input_ids"].shape[1])
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        generation_kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_logits": True,
            "output_scores": True,
            "stop_strings": ["</answer>"],
            "tokenizer": tokenizer,
        }
        if do_sample:
            generation_kwargs["temperature"] = 1.0
        hidden_steps: list[torch.Tensor] = []

        def capture_hidden(_module, _args, output) -> None:
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, tuple) and output:
                hidden = output[0]
            if hidden is None or hidden.ndim != 3:
                raise RuntimeError("language model did not expose its last hidden state")
            hidden_steps.append(hidden[:, -1, :].detach().float().cpu())

        handle = None
        if do_sample:
            handle = self._semantic_embedding_module().register_forward_hook(capture_hidden)
        try:
            outputs = self.model.generate(**inputs, **generation_kwargs)
        finally:
            if handle is not None:
                handle.remove()
        if outputs.logits is None or outputs.scores is None:
            raise RuntimeError("Transformers generation did not return token scores")
        step_count = len(outputs.logits)
        generated = outputs.sequences[:, prompt_width : prompt_width + step_count]
        raw_log_probs = torch.stack(
            [
                torch.log_softmax(step_logits.float(), dim=-1)
                .gather(1, generated[:, step].to(step_logits.device).unsqueeze(-1))
                .squeeze(-1)
                for step, step_logits in enumerate(outputs.logits)
            ],
            dim=1,
        )
        sampling_log_probs = torch.stack(
            [
                torch.log_softmax(step_scores.float(), dim=-1)
                .gather(1, generated[:, step].to(step_scores.device).unsqueeze(-1))
                .squeeze(-1)
                for step, step_scores in enumerate(outputs.scores)
            ],
            dim=1,
        )
        pad_id = tokenizer.pad_token_id
        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_ids = set(eos_ids or [])
        values: dict[str, GeneratedResponse] = {}
        for index, request in enumerate(requests):
            ids = [int(value) for value in generated[index].tolist()]
            length = len(ids)
            for position, token_id in enumerate(ids):
                if token_id in eos_ids:
                    length = position + 1
                    break
                if pad_id is not None and token_id == pad_id:
                    length = position
                    break
            ids = ids[:length]
            text = self.processor.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            values[request.request_id] = GeneratedResponse(
                text=text,
                token_ids=tuple(ids),
                token_log_probs=tuple(raw_log_probs[index, :length].cpu().tolist()),
                sampling_token_log_probs=tuple(
                    sampling_log_probs[index, :length].cpu().tolist()
                ),
                finish_reason="length" if length == max_new_tokens else "stop",
                rng_seed=batch_seed,
                hidden_steps=tuple(
                    tuple(float(value) for value in step[index].tolist())
                    for step in hidden_steps
                ),
            )
        return values
