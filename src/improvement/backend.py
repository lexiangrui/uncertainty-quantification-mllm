"""Model backend for ECA input construction.

Loads a multimodal model per family (LLaVA-1.5 / Qwen2.5-VL / InternVL3.5)
and builds teacher-forcing inputs: prompt + re-tokenized greedy response,
with XML section character spans mapped to token positions.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.generation.parser import section_character_spans
from src.generation.prompt import build_prompt


class EcaBackend:
    """Loads a multimodal model and builds teacher-forcing inputs for ECA.

    ECA needs attention weights, so attention must return them — pass
    ``attn_implementation="eager"`` (the default here).
    """

    def __init__(
        self,
        family: str,
        model_path: Path,
        *,
        adapter_path: Path | None = None,
        attn_implementation: str = "eager",
    ):
        self.family = family
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.attn_implementation = attn_implementation
        self.processor = None
        self.model = None
        self.device = None
        self._image_token_id: int | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self):
        if self.model is not None:
            return

        if self.family == "llava_1_5":
            from transformers import LlavaForConditionalGeneration
            cls = LlavaForConditionalGeneration
        elif self.family == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration
            cls = Qwen2_5_VLForConditionalGeneration
        elif self.family == "internvl3_5":
            from transformers import InternVLForConditionalGeneration
            cls = InternVLForConditionalGeneration
        else:
            raise ValueError(f"unsupported family: {self.family}")

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)

        dtype = torch.float16 if self.family == "llava_1_5" else torch.bfloat16
        device_map = "auto"

        kwargs = dict(
            device_map=device_map,
            low_cpu_mem_usage=True,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=self.attn_implementation,
        )
        self.model = cls.from_pretrained(self.model_path, **kwargs).eval()

        if self.adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path, local_files_only=True).eval()

        self.device = next(self.model.parameters()).device

        # Find image token id
        base_config = self._base_model().config
        for attr in ("image_token_index", "image_token_id"):
            val = getattr(base_config, attr, None)
            if val is not None:
                self._image_token_id = int(val)
                break

    def _base_model(self):
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_prompt(self, image, question):
        prompt = build_prompt(question, image is not None)
        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt.user})
        messages = [{"role": "user", "content": content}]
        if prompt.system:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": prompt.system}]})
        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        kwargs = {"text": rendered, "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = image
        inputs = self.processor(**kwargs)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def _section_token_spans(
        self, raw_response: str, char_spans: dict[str, tuple[int, int]]
    ) -> tuple[list[int], dict[str, tuple[int, int]]]:
        """Map XML content to tokens wholly contained in each character span."""
        tok = getattr(self.processor, "tokenizer", self.processor)
        if not getattr(tok, "is_fast", False):
            raise ValueError("ECA requires a fast tokenizer with offset mapping")
        encoded = tok(raw_response, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        spans = {}
        for name, (cs, ce) in char_spans.items():
            hits = [i for i, (start, end) in enumerate(offsets) if cs <= start < end <= ce]
            if not hits:
                raise ValueError(f"section has no XML-free tokens: {name}")
            spans[name] = (hits[0], hits[-1] + 1)
        return list(encoded["input_ids"]), spans

    def prepare_inputs_sections(self, image, question, raw_response: str):
        """Teacher-forcing inputs plus absolute token spans of all sections.

        Returns (full_inputs, prompt_length,
                 {section: (abs_start, abs_end)}) or (None, None, None) when
        the response has no separable sections.
        """
        prompt_inputs = self._prepare_prompt(image, question)
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        try:
            char_spans = section_character_spans(raw_response, "xml")
        except ValueError:
            return None, None, None
        gen_ids, tok_spans = self._section_token_spans(raw_response, char_spans)

        gen_tensor = torch.tensor([gen_ids], dtype=torch.long, device=self.device)
        full_ids = torch.cat([prompt_inputs["input_ids"], gen_tensor], dim=1)
        full_mask = torch.cat([prompt_inputs["attention_mask"], torch.ones_like(gen_tensor)], dim=1)

        full_inputs = dict(prompt_inputs)
        full_inputs["input_ids"] = full_ids
        full_inputs["attention_mask"] = full_mask

        # Qwen2.5-VL: extend mm_token_type_ids to match full sequence length
        for key in list(full_inputs.keys()):
            if key in ("input_ids", "attention_mask", "pixel_values"):
                continue
            val = full_inputs[key]
            if isinstance(val, torch.Tensor) and val.ndim >= 2 and val.shape[1] == prompt_length:
                # Extend with zeros (text modality) for generated tokens
                pad_shape = list(val.shape)
                pad_shape[1] = full_ids.shape[1] - prompt_length
                pad = torch.zeros(pad_shape, dtype=val.dtype, device=val.device)
                full_inputs[key] = torch.cat([val, pad], dim=1)

        abs_spans = {
            name: (prompt_length + ts, prompt_length + te)
            for name, (ts, te) in tok_spans.items()
        }
        return full_inputs, prompt_length, abs_spans

    def prepare_inputs(self, image, question, raw_response: str):
        """Build teacher-forcing inputs: prompt + generated response."""
        full_inputs, prompt_length, spans = self.prepare_inputs_sections(image, question, raw_response)
        if full_inputs is None:
            return None, None, None
        return full_inputs, prompt_length, spans["answer"]
