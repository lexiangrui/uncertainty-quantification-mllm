from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.generation.prompt import GenerationPrompt

from .base import GeneratedResponse, GenerationBackend


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
        if family == "internvl3_5":
            from transformers import AutoProcessor, InternVLForConditionalGeneration

            model_class = InternVLForConditionalGeneration
        elif family == "qwen2_5_vl":
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            model_class = Qwen2_5_VLForConditionalGeneration
        elif family == "llava_1_5":
            from transformers import AutoProcessor, LlavaForConditionalGeneration

            model_class = LlavaForConditionalGeneration
        else:
            raise ValueError(f"unknown model family: {family}")

        self.family = family
        self.model_id = model_path.name
        self.model_path = model_path
        self.model_class = model_class
        self.processor_class = AutoProcessor
        self.attn_implementation = attn_implementation
        self.adapter_path = adapter_path
        self.processor = None
        self.model = None
        self.device = None

    @property
    def runtime_config(self) -> dict:
        return {
            "attn_implementation": self.attn_implementation,
            "adapter_path": str(self.adapter_path) if self.adapter_path else None,
            "local_files_only": True,
        }

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
        self.processor = self.processor_class.from_pretrained(
            self.model_path, local_files_only=True
        )
        model_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        model_kwargs["dtype"] = (
            torch.float16 if self.family == "llava_1_5" else torch.bfloat16
        )
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self.model = self.model_class.from_pretrained(
            self.model_path, **model_kwargs
        ).eval()
        if self.adapter_path is not None:
            if not self.adapter_path.is_dir():
                raise NotADirectoryError(self.adapter_path)
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, self.adapter_path, local_files_only=True
            ).eval()
        self.device = next(self.model.parameters()).device

    def _inputs(self, image: Image.Image | None, prompt: GenerationPrompt):
        self._load()
        assert self.processor is not None and self.device is not None
        content: list[dict] = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt.user})
        messages = [{"role": "user", "content": content}]
        if prompt.system:
            messages.insert(
                0,
                {"role": "system", "content": [{"type": "text", "text": prompt.system}]},
            )
        template_kwargs = {"add_generation_prompt": True, "tokenize": False}
        rendered = self.processor.apply_chat_template(messages, **template_kwargs)
        kwargs = {"text": rendered, "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = image
        inputs = self.processor(**kwargs)
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

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
    def generate(
        self,
        image: Image.Image | None,
        prompt: GenerationPrompt,
        *,
        do_sample: bool,
        temperature: float | None,
        max_new_tokens: int,
        num_return_sequences: int,
    ) -> list[GeneratedResponse]:
        if do_sample != (temperature is not None):
            raise ValueError("temperature must be set exactly when sampling")
        if not do_sample and num_return_sequences != 1:
            raise ValueError("greedy generation requires one return sequence")
        inputs = self._inputs(image, prompt)
        assert self.model is not None and self.processor is not None
        prompt_length = inputs["input_ids"].shape[1]
        kwargs = {
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_logits": True,
            "output_scores": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        latest_hidden = None

        def capture_last_hidden(_module, _args, output):
            nonlocal latest_hidden
            hidden = getattr(output, "last_hidden_state", None)
            if hidden is None and isinstance(output, tuple) and output:
                hidden = output[0]
            if hidden is None or hidden.ndim != 3:
                raise RuntimeError("language model did not expose its last hidden state")
            latest_hidden = hidden[:, -1, :].detach()

        handle = self._semantic_embedding_module().register_forward_hook(
            capture_last_hidden
        )
        try:
            outputs = self.model.generate(**inputs, **kwargs)
        finally:
            handle.remove()
        if latest_hidden is None:
            raise RuntimeError("generation did not produce a response embedding")
        if outputs.logits is None:
            raise RuntimeError("model generation did not return raw logits")
        if outputs.scores is None:
            raise RuntimeError("model generation did not return sampling scores")
        step_count = len(outputs.logits)
        generated = outputs.sequences[:, prompt_length : prompt_length + step_count]
        token_log_probs = torch.stack(
            [
                torch.log_softmax(step_logits.float(), dim=-1)
                .gather(1, generated[:, step].to(step_logits.device).unsqueeze(-1))
                .squeeze(-1)
                for step, step_logits in enumerate(outputs.logits)
            ],
            dim=1,
        )
        sampling_token_log_probs = torch.stack(
            [
                torch.log_softmax(step_scores.float(), dim=-1)
                .gather(1, generated[:, step].to(step_scores.device).unsqueeze(-1))
                .squeeze(-1)
                for step, step_scores in enumerate(outputs.scores)
            ],
            dim=1,
        )
        texts = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if len(texts) != num_return_sequences:
            raise RuntimeError(
                f"model returned {len(texts)} sequences; expected {num_return_sequences}"
            )
        responses = []
        for index, text in enumerate(texts):
            ids = generated[index].tolist()
            responses.append(
                GeneratedResponse(
                    text=text.strip(),
                    token_ids=tuple(ids),
                    token_log_probs=tuple(token_log_probs[index].cpu().tolist()),
                    sampling_token_log_probs=tuple(
                        sampling_token_log_probs[index].cpu().tolist()
                    ),
                    final_hidden=tuple(latest_hidden[index].float().cpu().tolist()),
                )
            )
        return responses
