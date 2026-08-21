"""Model backend for ERA (Early Rationale Attribution) input construction."""
from __future__ import annotations

from pathlib import Path

import torch

from src.generation.prompt import build_prompt
from src.models.internvl import INTERNVL_SYSTEM_PROMPT, dynamic_image_tiles
from src.models.transformers_compat import patch_tied_weights_keys_compat


class EraBackend:
    """Loads a multimodal model and builds teacher-forcing inputs for ERA.

    ERA needs attention weights, so attention must return them — pass
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
        elif self.family == "internvl3_5_original":
            from transformers import AutoModel, AutoTokenizer

            patch_tied_weights_keys_compat()

            self.processor = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False,
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                use_flash_attn=False,
            ).eval()
            if self.adapter_path is not None:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(
                    self.model, self.adapter_path, local_files_only=True
                ).eval()
            self.device = next(self.model.parameters()).device
            self._image_token_id = int(
                self.processor.convert_tokens_to_ids("<IMG_CONTEXT>")
            )
            return
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
        if self.family == "internvl3_5_original":
            if image is None:
                raise ValueError("original InternVL ERA requires an image")
            from torchvision.transforms import InterpolationMode
            import torchvision.transforms as T

            base_model = self._base_model()
            config = base_model.config
            image_size = int(getattr(config, "force_image_size", None) or 448)
            transform = T.Compose(
                [
                    T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            image_tiles = dynamic_image_tiles(image, config)
            pixel_values = torch.stack([transform(tile) for tile in image_tiles]).to(
                self.device, dtype=torch.bfloat16
            )
            num_image_tokens = int(base_model.num_image_token)
            image_tokens = (
                "<img>"
                + "<IMG_CONTEXT>" * (num_image_tokens * len(image_tiles))
                + "</img>"
            )
            user = f"<image>\n{prompt.user}".replace("<image>", image_tokens, 1)
            rendered = (
                f"<|im_start|>system\n{INTERNVL_SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            self.processor.padding_side = "left"
            inputs = self.processor(
                rendered, return_tensors="pt", add_special_tokens=False
            )
            inputs["pixel_values"] = pixel_values
            inputs["image_flags"] = torch.ones(
                (len(image_tiles), 1), dtype=torch.long, device=self.device
            )
            base_model.img_context_token_id = self._image_token_id
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
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
    ) -> tuple[list[int], list[int]]:
        """Align exact generated IDs into 3 continuous section slices (Vision, Reasoning, Answer).

        Buckets:
          2 = vision (<vision>...</vision>)
          3 = reasoning (<reasoning>...</reasoning>)
          4 = answer (<answer>...</answer>)

        XML tags are retained inside their respective sections for 100% sample stability
        and zero short-answer loss.
        """
        tok = getattr(self.processor, "tokenizer", self.processor)
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

        # Find character positions of section starts in raw_response
        r_char_pos = raw_response.lower().find("<reasoning")
        a_char_pos = raw_response.lower().find("<answer")
        if r_char_pos <= 0 or a_char_pos <= r_char_pos:
            raise ValueError("raw_response does not contain valid <reasoning> and <answer> tags")

        # Find exact token indices where each section begins via cumulative prefix decoding
        r_start_idx = None
        a_start_idx = None
        for i in range(1, len(gen_ids) + 1):
            prefix = tok.decode(
                gen_ids[:i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            prefix_len = len(prefix.lstrip())
            if r_start_idx is None and prefix_len > r_char_pos:
                r_start_idx = i - 1
            if a_start_idx is None and prefix_len > a_char_pos:
                a_start_idx = i - 1
                break

        if r_start_idx is None or a_start_idx is None or r_start_idx >= a_start_idx or r_start_idx == 0:
            raise ValueError("could not locate valid section boundaries in token sequence")

        # 2=vision [0, r_start_idx), 3=reasoning [r_start_idx, a_start_idx), 4=answer [a_start_idx, len]
        buckets = [2] * len(gen_ids)
        for i in range(r_start_idx, a_start_idx):
            buckets[i] = 3
        for i in range(a_start_idx, len(gen_ids)):
            buckets[i] = 4
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
        gen_ids, generated_buckets = self._align_generated_tokens(
            raw_response, generated_token_ids
        )

        gen_tensor = torch.tensor([gen_ids], dtype=torch.long, device=self.device)
        full_ids = torch.cat([prompt_inputs["input_ids"], gen_tensor], dim=1)
        full_mask = torch.cat([prompt_inputs["attention_mask"], torch.ones_like(gen_tensor)], dim=1)

        full_inputs = dict(prompt_inputs)
        full_inputs["input_ids"] = full_ids
        full_inputs["attention_mask"] = full_mask
        # Pop position_ids so multimodal architectures (e.g. Qwen2.5-VL 3D RoPE)
        # automatically recalculate correct full-sequence position IDs
        full_inputs.pop("position_ids", None)

        # Qwen2.5-VL / multimodal: extend mm_token_type_ids to match full sequence length
        for key in list(full_inputs.keys()):
            if key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
                continue
            val = full_inputs[key]
            if isinstance(val, torch.Tensor) and val.ndim >= 2 and val.shape[-1] == prompt_length:
                # Extend with zeros (text modality) for generated tokens along sequence dimension
                pad_shape = list(val.shape)
                pad_shape[-1] = full_ids.shape[1] - prompt_length
                pad = torch.zeros(pad_shape, dtype=val.dtype, device=val.device)
                full_inputs[key] = torch.cat([val, pad], dim=-1)

        return full_inputs, prompt_length, generated_buckets
