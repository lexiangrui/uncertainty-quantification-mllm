"""Model backend for ECA input construction."""
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

    def _align_generated_tokens(
        self,
        raw_response: str,
        generated_token_ids: list[int],
        char_spans: dict[str, tuple[int, int]],
    ) -> tuple[list[int], list[int]]:
        """Align exact generated IDs to semantic sections without re-tokenizing.

        Buckets are 2=vision, 3=reasoning, 4=answer, and 5=XML/whitespace.
        A token crossing a semantic boundary stays in bucket 5.  Exact ID
        equality with a single full-response encoding is required so the
        teacher-forced sequence is the sequence produced during generation.
        """
        tok = getattr(self.processor, "tokenizer", self.processor)
        if not getattr(tok, "is_fast", False):
            raise ValueError("ECA alignment requires a fast tokenizer")

        gen_ids = list(generated_token_ids)
        eos_ids = getattr(tok, "eos_token_id", None)
        terminal_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
        pad_id = getattr(tok, "pad_token_id", None)
        terminal_ids.discard(None)
        if pad_id is not None:
            terminal_ids.add(pad_id)
        while gen_ids and gen_ids[-1] in terminal_ids:
            gen_ids.pop()
        if not gen_ids:
            raise ValueError("generated response has no text tokens")

        decoded = tok.decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded.strip() != raw_response:
            raise ValueError("generated token IDs do not decode to raw_response")
        response_start = len(decoded) - len(decoded.lstrip())
        encoded = tok(
            decoded,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if list(encoded["input_ids"]) != gen_ids:
            raise ValueError("generated token IDs are not stable under full-response encoding")

        shifted = {
            name: (response_start + start, response_start + end)
            for name, (start, end) in char_spans.items()
        }
        buckets = [5] * len(gen_ids)
        bucket_ids = {"vision": 2, "reasoning": 3, "answer": 4}
        for index, (start, end) in enumerate(encoded["offset_mapping"]):
            if start == end:
                continue
            for name, (section_start, section_end) in shifted.items():
                if section_start <= start and end <= section_end:
                    buckets[index] = bucket_ids[name]
                    break
        for name, bucket in bucket_ids.items():
            if bucket not in buckets:
                raise ValueError(f"{name} has no XML-free generated token")
        return gen_ids, buckets

    def prepare_inputs_sections(
        self,
        image,
        question,
        raw_response: str,
        generated_token_ids: list[int],
    ):
        """Build teacher-forcing inputs from the exact generated token IDs."""
        prompt_inputs = self._prepare_prompt(image, question)
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        char_spans = section_character_spans(raw_response, "xml")
        gen_ids, generated_buckets = self._align_generated_tokens(
            raw_response, generated_token_ids, char_spans
        )

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

        return full_inputs, prompt_length, generated_buckets
