from __future__ import annotations

import gc
import os
from pathlib import Path

import torch
from PIL import Image

from src.generation.prompt import GenerationPrompt

from .base import GeneratedResponse, GenerationRequest
from .internvl import INTERNVL_SYSTEM_PROMPT, dynamic_image_tiles
from .transformers_compat import patch_tied_weights_keys_compat


class HuggingFaceReplayBackend:
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
        if family not in {"internvl3_5", "internvl3_5_original", "qwen2_5_vl", "llava_1_5"}:
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
        self._replay_batch_limit: int | None = None

    @property
    def runtime_config(self) -> dict:
        config = {
            "engine": "hf_replay",
            "attn_implementation": self.attn_implementation,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "local_files_only": True,
            "adaptive_oom_split": True,
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

        if self.attn_implementation == "flash_attention_2":
            from transformers.utils import is_flash_attn_2_available

            if not torch.cuda.is_available() or not is_flash_attn_2_available():
                raise RuntimeError(
                    "FlashAttention2 requires an initialized CUDA device and the local "
                    "flash-attn extension. The current Slurm allocation has no usable CUDA "
                    "device; resubmit on a healthy GPU node instead of falling back to a "
                    "Hub kernel or CPU attention."
                )

        if self.family == "internvl3_5_original":
            from transformers import AutoModel, AutoTokenizer

            patch_tied_weights_keys_compat()
            self.processor = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False,
            )
            model_kwargs = {
                "device_map": "auto",
                "low_cpu_mem_usage": True,
                "local_files_only": True,
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "use_flash_attn": self.attn_implementation == "flash_attention_2",
            }
            self.model = AutoModel.from_pretrained(self.model_path, **model_kwargs).eval()
        elif self.family == "internvl3_5":
            from transformers import InternVLForConditionalGeneration as model_class
        elif self.family == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration as model_class
        elif self.family == "llava_1_5":
            from transformers import LlavaForConditionalGeneration as model_class
        else:
            raise ValueError(f"unknown model family: {self.family}")

        if self.family != "internvl3_5_original":
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
        if self.attn_implementation == "flash_attention_2" and self.family != "internvl3_5_original":
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

    def _batch_inputs(self, requests: list[GenerationRequest]):
        self._load()
        assert self.processor is not None and self.device is not None
        if self.family == "internvl3_5_original":
            return self._original_batch_inputs(requests)
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

    def _semantic_embedding_module(self):
        """Return the decoder body whose output is the final hidden layer.

        Replay only needs the answer-last-token representation for sampled
        responses. Hooking the decoder body avoids materializing every layer's
        hidden state for the full multimodal sequence.
        """
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
        decoder = getattr(language_model, "model", None)
        return decoder if decoder is not None else language_model

    def _original_batch_inputs(self, requests: list[GenerationRequest]):
        """Build the original InternVLChatModel inputs (image tiles + IMG_CONTEXT)."""
        assert self.model is not None and self.processor is not None and self.device is not None
        has_images = {request.image is not None for request in requests}
        if len(has_images) != 1:
            raise ValueError("an original InternVL batch cannot mix image and text-only requests")
        tokenizer = self.processor
        tokenizer.padding_side = "left"
        if has_images == {False}:
            rendered = [
                f"<|im_start|>system\n{request.prompt.system or INTERNVL_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{request.prompt.user}<|im_end|>\n"
                "<|im_start|>assistant\n"
                for request in requests
            ]
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            return {name: value.to(self.device) for name, value in inputs.items()}

        from PIL import Image
        from torchvision.transforms import InterpolationMode
        import torchvision.transforms as T

        base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        config = base_model.config
        image_size = int(getattr(config, "force_image_size", None) or 448)
        transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        num_image_tokens = int(base_model.num_image_token)
        rendered: list[str] = []
        tiles: list[torch.Tensor] = []
        image_flags: list[torch.Tensor] = []
        for request in requests:
            image = request.image
            assert image is not None
            if not isinstance(image, Image.Image):
                raise TypeError("original InternVL expects PIL images")
            image_tiles = dynamic_image_tiles(image, config)
            tiles.append(torch.stack([transform(tile) for tile in image_tiles]))
            image_flags.append(torch.ones((len(image_tiles), 1), dtype=torch.long))
            question = f"<image>\n{request.prompt.user}"
            image_tokens = (
                "<img>"
                + "<IMG_CONTEXT>" * (num_image_tokens * len(image_tiles))
                + "</img>"
            )
            question = question.replace("<image>", image_tokens, 1)
            rendered.append(
                f"<|im_start|>system\n{request.prompt.system or INTERNVL_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        inputs = tokenizer(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        inputs["pixel_values"] = torch.cat(tiles, dim=0).to(self.device, dtype=torch.bfloat16)
        inputs["image_flags"] = torch.cat(image_flags, dim=0).to(self.device)
        base_model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        return {name: value.to(self.device) for name, value in inputs.items()}

    def _replay_forward(self, full_inputs: dict, *, logits_to_keep: int):
        assert self.model is not None
        kwargs = {
            "use_cache": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if self.family == "internvl3_5_original":
            base_model = (
                self.model.get_base_model()
                if hasattr(self.model, "get_base_model")
                else self.model
            )
            if "pixel_values" not in full_inputs:
                return base_model.language_model(
                    **full_inputs, **kwargs, logits_to_keep=logits_to_keep
                )

            # InternVL's outer forward does not expose Qwen's logits_to_keep.
            # Inject it at the language-model boundary while retaining the
            # upstream multimodal embedding path byte-for-byte.
            def limit_logits(_module, args, model_kwargs):
                model_kwargs["logits_to_keep"] = logits_to_keep
                return args, model_kwargs

            handle = base_model.language_model.register_forward_pre_hook(
                limit_logits, with_kwargs=True
            )
            try:
                return self.model(**full_inputs, **kwargs)
            finally:
                handle.remove()
        return self.model(**full_inputs, **kwargs)

    @torch.inference_mode()
    def teacher_force_responses(
        self,
        requests: list[GenerationRequest],
        token_sequences: list[tuple[int, ...]],
    ) -> dict[str, GeneratedResponse]:
        """Replay exact vLLM tokens, splitting only when the GPU reports OOM."""
        if len(requests) != len(token_sequences):
            raise ValueError("replay requests and token sequences must have equal length")
        if not requests:
            return {}
        for request, token_ids in zip(requests, token_sequences, strict=True):
            if not token_ids:
                raise ValueError(f"empty replay token sequence: {request.request_id}")
        batch_limit = getattr(self, "_replay_batch_limit", None)
        if batch_limit is not None and len(requests) > batch_limit:
            values: dict[str, GeneratedResponse] = {}
            for start in range(0, len(requests), batch_limit):
                stop = start + batch_limit
                values.update(
                    self.teacher_force_responses(
                        requests[start:stop], token_sequences[start:stop]
                    )
                )
            return values
        try:
            return self._teacher_force_batch(requests, token_sequences)
        except torch.OutOfMemoryError:
            if len(requests) == 1:
                raise

        # Retry only after leaving the exception handler. Otherwise Python keeps
        # the failed forward traceback (and its CUDA tensors) alive while the
        # smaller recursive batches run, so each retry has less free memory.
        learned_limit = max(1, len(requests) // 2)
        current_limit = getattr(self, "_replay_batch_limit", None)
        self._replay_batch_limit = (
            learned_limit
            if current_limit is None
            else min(current_limit, learned_limit)
        )
        print(
            f"HF replay OOM at batch_size={len(requests)}; "
            f"retrying with persistent batch_limit={self._replay_batch_limit}",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return self.teacher_force_responses(requests, token_sequences)

    @torch.inference_mode()
    def _teacher_force_batch(
        self,
        requests: list[GenerationRequest],
        token_sequences: list[tuple[int, ...]],
    ) -> dict[str, GeneratedResponse]:
        inputs = self._batch_inputs(requests)
        assert self.model is not None and self.processor is not None
        prompt_width = int(inputs["input_ids"].shape[1])
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            eos_id = tokenizer.eos_token_id
            pad_id = eos_id[0] if isinstance(eos_id, list) else eos_id
        if pad_id is None:
            pad_id = 0
        max_generated = max(len(ids) for ids in token_sequences)
        generated = torch.full(
            (len(requests), max_generated),
            int(pad_id),
            dtype=torch.long,
            device=self.device,
        )
        generated_mask = torch.zeros_like(generated)
        for index, token_ids in enumerate(token_sequences):
            length = len(token_ids)
            generated[index, :length] = torch.tensor(
                token_ids, dtype=torch.long, device=self.device
            )
            generated_mask[index, :length] = 1
        full_inputs = dict(inputs)
        full_inputs["input_ids"] = torch.cat([inputs["input_ids"], generated], dim=1)
        if "attention_mask" in inputs:
            full_inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], generated_mask], dim=1
            )
        full_inputs.pop("position_ids", None)
        for key, value in list(full_inputs.items()):
            if key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}:
                continue
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                continue
            if value.shape[-1] != prompt_width:
                continue
            padding = torch.zeros(
                (*value.shape[:-1], max_generated), dtype=value.dtype, device=value.device
            )
            full_inputs[key] = torch.cat([value, padding], dim=-1)
        roles = {request.role for request in requests}
        if len(roles) != 1:
            raise ValueError("HF replay batches must contain one decoding role")
        need_hidden = roles == {"sample"}
        captured_hidden: list[torch.Tensor] = []
        handle = None
        if need_hidden:
            def capture_hidden(_module, _args, output) -> None:
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None and isinstance(output, tuple) and output:
                    hidden = output[0]
                if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                    raise RuntimeError("language decoder did not expose its last hidden state")
                captured_hidden.append(
                    hidden[:, -max_generated:, :]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                )

            handle = self._semantic_embedding_module().register_forward_hook(
                capture_hidden
            )
        try:
            outputs = self._replay_forward(
                full_inputs, logits_to_keep=max_generated + 1
            )
        finally:
            if handle is not None:
                handle.remove()
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise RuntimeError("HF replay model did not return logits")
        logits_offset = int(full_inputs["input_ids"].shape[1] - logits.shape[1])
        first_answer_logit = prompt_width - 1 - logits_offset
        if first_answer_logit < 0:
            raise RuntimeError("HF replay returned too few logits for answer scoring")
        last_hidden = captured_hidden[-1] if captured_hidden else None
        if need_hidden and last_hidden is None:
            raise RuntimeError("HF replay did not capture the final hidden layer")
        values: dict[str, GeneratedResponse] = {}
        for index, (request, token_ids) in enumerate(
            zip(requests, token_sequences, strict=True)
        ):
            length = len(token_ids)
            positions = slice(first_answer_logit, first_answer_logit + length)
            selected = logits[index, positions, :].float()
            targets = generated[index, :length].to(selected.device)
            token_log_probs = (
                selected.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                - torch.logsumexp(selected, dim=-1)
            )
            hidden_steps = None
            if last_hidden is not None:
                hidden_steps = last_hidden[index, :length, :]
            log_probs = tuple(float(value) for value in token_log_probs.cpu().tolist())
            values[request.request_id] = GeneratedResponse(
                text=self.processor.decode(
                    list(token_ids),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip(),
                token_ids=token_ids,
                token_log_probs=log_probs,
                sampling_token_log_probs=log_probs,
                hidden_steps=hidden_steps,
                finish_reason="stop",
                rng_seed=request.seed,
            )
        return values
