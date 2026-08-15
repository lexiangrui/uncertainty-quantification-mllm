"""Model backend for SAI — model loading and teacher-forcing input construction.

Loads a multimodal model per family (LLaVA-1.5 / Qwen2.5-VL / InternVL3.5)
and builds teacher-forcing inputs: prompt + re-tokenized greedy response.
Beyond the shared pipeline this backend exposes:

* character-span → token-position mapping for arbitrary response spans
  (object mentions inside the vision/reasoning/answer sections);
* visual token position detection via the family image-token id;
* the language-model head / final norm used by the vision logit lens.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.generation.parser import answer_character_span
from src.generation.prompt import build_prompt


@dataclass
class TeacherForcingBatch:
    """A prepared teacher-forcing input plus span/position metadata."""

    inputs: dict
    prompt_length: int
    gen_ids: list[int]
    offsets: list[tuple[int, int]] | None  # char span per generated token

    def char_span_tokens(self, char_start: int, char_end: int) -> list[int]:
        """Generated-token indices overlapping [char_start, char_end).

        Byte-level BPE sometimes merges the closing '>' of an opening XML
        tag into the following token; a token belongs to the span when it
        *overlaps* it, matching the VGS answer-span convention.
        """
        if self.offsets is not None:
            return [
                i
                for i, (s, e) in enumerate(self.offsets)
                if e > char_start and s < char_end
            ]
        # Slow-tokenizer fallback: decode-prefix cover, mirroring VGS.
        tok = self._tok
        tok_start = tok_end = None
        for end in range(1, len(self.gen_ids) + 1):
            decoded = tok.decode(
                self.gen_ids[:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if tok_start is None and len(decoded) > char_start:
                tok_start = end - 1
            if len(decoded) >= char_end:
                tok_end = end
                break
        if tok_start is None:
            return []
        if tok_end is None:
            tok_end = len(self.gen_ids)
        return list(range(tok_start, tok_end))

    _tok: object = None

    def absolute(self, gen_idx: int) -> int:
        """Absolute sequence position of generated-token index ``gen_idx``."""
        return self.prompt_length + gen_idx


class SaiBackend:
    """Loads a multimodal model and builds teacher-forcing inputs for SAI."""

    def __init__(
        self,
        family: str,
        model_path: Path,
        *,
        adapter_path: Path | None = None,
        attn_implementation: str = "sdpa",
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

        # bfloat16 for every family: SAI measures within-run deltas
        # (baseline vs intervened forward), so the wider dynamic range of
        # bf16 (large late-layer rotations overflow fp16) matters more than
        # matching the fp16 generation-time logits exactly.
        dtype = torch.bfloat16
        kwargs = dict(
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=self.attn_implementation,
        )
        self.model = cls.from_pretrained(self.model_path, **kwargs).eval()

        if self.adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, self.adapter_path, local_files_only=True
            ).eval()

        self.device = next(self.model.parameters()).device

        base_config = self._base_model().config
        for attr in ("image_token_index", "image_token_id"):
            val = getattr(base_config, attr, None)
            if val is not None:
                self._image_token_id = int(val)
                break

    def _base_model(self):
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    # ------------------------------------------------------------------
    # LM head / final norm (for the vision logit lens)
    # ------------------------------------------------------------------

    def lm_head_weight(self) -> torch.Tensor:
        """Unembedding matrix W (vocab × hidden) of the language-model head."""
        base = self._base_model()
        head = getattr(base, "lm_head", None)
        if head is None:
            core = getattr(base, "model", base)
            lang = getattr(core, "language_model", core)
            head = getattr(lang, "lm_head", None)
        if head is None:
            raise RuntimeError("cannot locate lm_head")
        return head.weight.data

    def final_norm(self):
        """The final RMSNorm applied to the residual stream before lm_head."""
        base = self._base_model()
        core = getattr(base, "model", base)
        lang = getattr(core, "language_model", core)
        decoder = getattr(lang, "model", lang)
        norm = getattr(decoder, "norm", None)
        if norm is None:
            # transformers >= 5: language_model itself may be the decoder.
            norm = getattr(lang, "norm", None)
        return norm

    def decoder_layers(self) -> list:
        base = self._base_model()
        core = getattr(base, "model", base)
        lang = getattr(core, "language_model", core)
        decoder = getattr(lang, "model", lang)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            raise RuntimeError("cannot locate decoder layers")
        return list(layers)

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        return getattr(self.processor, "tokenizer", self.processor)

    def _prepare_prompt(self, image, question):
        prompt = build_prompt(question, image is not None)
        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt.user})
        messages = [{"role": "user", "content": content}]
        if prompt.system:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": prompt.system}]})
        rendered = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        kwargs = {"text": rendered, "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = image
        inputs = self.processor(**kwargs)
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    def prepare_inputs(self, image, question, raw_response: str) -> TeacherForcingBatch | None:
        """Build teacher-forcing inputs: prompt + generated response."""
        prompt_inputs = self._prepare_prompt(image, question)
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        tok = self.tokenizer

        offsets = None
        gen_ids = tok.encode(raw_response, add_special_tokens=False)
        if getattr(tok, "is_fast", False):
            enc = tok(raw_response, add_special_tokens=False, return_offsets_mapping=True)
            gen_ids = list(enc["input_ids"])
            offsets = [tuple(o) for o in enc["offset_mapping"]]
        if not gen_ids:
            return None
        # Sanity: the answer section must be locatable.
        try:
            answer_character_span(raw_response, "xml")
        except ValueError:
            return None

        gen_tensor = torch.tensor([gen_ids], dtype=torch.long, device=self.device)
        full_ids = torch.cat([prompt_inputs["input_ids"], gen_tensor], dim=1)
        full_mask = torch.cat([prompt_inputs["attention_mask"], torch.ones_like(gen_tensor)], dim=1)

        full_inputs = dict(prompt_inputs)
        full_inputs["input_ids"] = full_ids
        full_inputs["attention_mask"] = full_mask

        for key in list(full_inputs.keys()):
            if key in ("input_ids", "attention_mask", "pixel_values"):
                continue
            val = full_inputs[key]
            if isinstance(val, torch.Tensor) and val.ndim >= 2 and val.shape[1] == prompt_length:
                pad_shape = list(val.shape)
                pad_shape[1] = full_ids.shape[1] - prompt_length
                pad = torch.zeros(pad_shape, dtype=val.dtype, device=val.device)
                full_inputs[key] = torch.cat([val, pad], dim=1)

        batch = TeacherForcingBatch(
            inputs=full_inputs, prompt_length=prompt_length, gen_ids=gen_ids, offsets=offsets
        )
        batch._tok = tok
        return batch

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------

    def visual_positions(self, inputs: dict) -> torch.Tensor:
        """Absolute positions of visual tokens in the merged sequence."""
        if self._image_token_id is None:
            raise RuntimeError("image token id unknown for this family")
        mask = inputs["input_ids"][0] == self._image_token_id
        return mask.nonzero(as_tuple=True)[0]
