"""Model backend for EPAR — model loading and teacher-forcing input construction.

Loads a multimodal model per family (LLaVA-1.5 / Qwen2.5-VL / InternVL3.5)
and builds teacher-forcing inputs: prompt + re-tokenized greedy response,
with the answer character span mapped to token positions.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.generation.parser import answer_character_span
from src.generation.prompt import build_prompt


class EparBackend:
    """Loads a multimodal model and builds teacher-forcing inputs for EPAR.

    EPAR needs attention weights, so attention must return them — pass
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

    def _tokenize(self, text: str) -> list[int]:
        tok = getattr(self.processor, "tokenizer", self.processor)
        return tok.encode(text, add_special_tokens=False)

    def _decode(self, token_ids: list[int]) -> str:
        tok = getattr(self.processor, "tokenizer", self.processor)
        return tok.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    def _answer_token_span(
        self, raw_response: str, gen_ids: list[int], char_start: int, char_end: int
    ) -> tuple[list[int], int, int]:
        """Map the answer character span to generated-token indices.

        Uses the fast tokenizer's offset mapping: a token belongs to the
        answer when it *overlaps* [char_start, char_end).  Byte-level BPE
        sometimes merges the closing '>' of <answer> (or the '<' of
        </answer>) into the first/last answer token; such a token is the
        answer's boundary token and is kept — there is no finer split.

        Returns (gen_ids, tok_start, tok_end) where gen_ids may be replaced
        by the re-encoding the offsets were computed against.
        """
        tok = getattr(self.processor, "tokenizer", self.processor)
        if getattr(tok, "is_fast", False):
            enc = tok(raw_response, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(enc["input_ids"])
            hits = [
                i for i, (s, e) in enumerate(enc["offset_mapping"])
                if e > char_start and s < char_end
            ]
            if hits:
                return ids, hits[0], hits[-1] + 1
            gen_ids = ids
        # Fallback (slow tokenizers): first token whose decoded prefix
        # covers the span.  Strict '>' so a token ending exactly at
        # char_start (e.g. the '>' of <answer>) is not included.
        tok_start = tok_end = None
        for end in range(1, len(gen_ids) + 1):
            decoded = self._decode(gen_ids[:end])
            if tok_start is None and len(decoded) > char_start:
                tok_start = end - 1
            if len(decoded) >= char_end:
                tok_end = end
                break
        if tok_start is None:
            tok_start = 0
        if tok_end is None:
            tok_end = len(gen_ids)
        return list(gen_ids), tok_start, tok_end

    def prepare_inputs(self, image, question, raw_response: str):
        """Build teacher-forcing inputs: prompt + generated response."""
        prompt_inputs = self._prepare_prompt(image, question)
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        gen_ids = self._tokenize(raw_response)
        if not gen_ids:
            return None, None, None

        # Find answer span within generated tokens
        try:
            char_start, char_end = answer_character_span(raw_response, "xml")
        except ValueError:
            return None, None, None
        # May replace gen_ids with the offset-consistent re-encoding
        gen_ids, tok_start, tok_end = self._answer_token_span(raw_response, gen_ids, char_start, char_end)

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

        return full_inputs, prompt_length, (prompt_length + tok_start, prompt_length + tok_end)
