"""Layer-wise Answer Consistency (LAC) — Logit Lens based UQ.

Core idea: project each transformer layer's hidden state to vocabulary
space (Logit Lens).  For confident hallucinations, intermediate layers
may disagree with the final output — the model might "know" the answer
is uncertain at some layers but the final layer produces a confident
wrong answer due to language-prior override.

A single teacher-forcing forward pass with ``output_hidden_states=True``
suffices; no perturbation or multiple sampling needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.generation.parser import answer_character_span
from src.generation.prompt import build_prompt


@dataclass
class LacResult:
    score: float | None
    base_nll: float | None
    early_nll: float | None       # mean NLL from Logit Lens at early layers
    late_nll: float | None        # mean NLL from Logit Lens at late layers
    nll_trajectory: list[float]   # per-layer NLL (length = n_layers)
    answer_token_count: int

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "base_nll": self.base_nll,
            "early_nll": self.early_nll,
            "late_nll": self.late_nll,
            "nll_trajectory": self.nll_trajectory,
            "answer_token_count": self.answer_token_count,
        }


class LacBackend:
    """Loads a multimodal model and computes Layer-wise Answer Consistency."""

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
        self._final_norm = None
        self._lm_head = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self):
        if self.model is not None:
            return

        if self.family == "llava_1_5":
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            cls = LlavaForConditionalGeneration
        elif self.family == "qwen2_5_vl":
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            cls = Qwen2_5_VLForConditionalGeneration
        elif self.family == "internvl3_5":
            from transformers import AutoProcessor, InternVLForConditionalGeneration
            cls = InternVLForConditionalGeneration
        else:
            raise ValueError(f"unsupported family: {self.family}")

        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)

        dtype = torch.float16 if self.family == "llava_1_5" else torch.bfloat16
        device_map = "auto"
        if self.family == "qwen2_5_vl":
            device_map = {"model.visual": 0, "model.language_model": 1, "lm_head": 1}

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

        # Cache final layer norm and lm_head for Logit Lens
        self._cache_projection_modules()

    def _base_model(self):
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def _cache_projection_modules(self):
        """Find the final layer norm and LM head for Logit Lens projection."""
        base = self._base_model()
        core = getattr(base, "model", base)
        lang_model = getattr(core, "language_model", core)
        decoder = getattr(lang_model, "model", lang_model)
        self._final_norm = getattr(decoder, "norm", None)
        if self._final_norm is None:
            self._final_norm = getattr(lang_model, "norm", None)
        self._lm_head = getattr(base, "lm_head", None)
        if self._lm_head is None:
            self._lm_head = getattr(lang_model, "lm_head", None)
        if self._final_norm is None or self._lm_head is None:
            raise RuntimeError("cannot locate final layer norm or lm_head")

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

    def prepare_inputs(self, image, question, raw_response: str):
        """Build teacher-forcing inputs: prompt + generated response."""
        prompt_inputs = self._prepare_prompt(image, question)
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        gen_ids = self._tokenize(raw_response)
        if not gen_ids:
            return None, None, None

        gen_tensor = torch.tensor([gen_ids], dtype=torch.long, device=self.device)
        full_ids = torch.cat([prompt_inputs["input_ids"], gen_tensor], dim=1)
        full_mask = torch.cat([prompt_inputs["attention_mask"], torch.ones_like(gen_tensor)], dim=1)

        # Find answer span within generated tokens
        try:
            char_start, char_end = answer_character_span(raw_response, "xml")
        except ValueError:
            return None, None, None

        tok_start = tok_end = None
        for end in range(1, len(gen_ids) + 1):
            decoded = self._decode(gen_ids[:end])
            if tok_start is None and len(decoded) >= char_start:
                tok_start = max(end - 1, 0)
            if len(decoded) >= char_end:
                tok_end = end
                break
        if tok_end is None:
            tok_end = len(gen_ids)
        if tok_start is None:
            tok_start = 0

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

    # ------------------------------------------------------------------
    # LAC computation
    # ------------------------------------------------------------------

    def compute_lac(self, full_inputs: dict, prompt_length: int, answer_span: tuple[int, int]) -> LacResult:
        """Run forward pass and compute layer-wise answer consistency."""
        input_ids = full_inputs["input_ids"]
        ans_start, ans_end = answer_span
        n_ans = ans_end - ans_start
        if n_ans < 1:
            return LacResult(None, None, None, None, [], 0)

        with torch.inference_mode():
            outputs = self.model(**full_inputs, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors
        n_layers = len(hidden_states) - 1  # subtract embedding output

        # Final-layer NLL (standard, same as PPL)
        logits = outputs.logits  # (1, seq, vocab)
        base_nll = self._compute_nll_from_logits(logits, input_ids, ans_start, ans_end)

        # Per-layer NLL via Logit Lens
        nll_trajectory: list[float] = []
        answer_ids = input_ids[0, ans_start:ans_end]  # (n_ans,)

        # Positions that PREDICT answer tokens: [ans_start-1, ans_end-1)
        predict_positions = list(range(ans_start - 1, ans_end - 1))

        for layer_idx in range(1, n_layers + 1):  # skip embedding output (idx 0)
            h = hidden_states[layer_idx][0, predict_positions]  # (n_ans, hidden_dim)
            h = h.to(self._final_norm.weight.device)
            h_norm = self._final_norm(h)  # (n_ans, hidden_dim)
            layer_logits = self._lm_head(h_norm)  # (n_ans, vocab)
            log_probs = F.log_softmax(layer_logits.float(), dim=-1)
            token_lps = log_probs.gather(-1, answer_ids.to(log_probs.device).unsqueeze(-1)).squeeze(-1)
            nll_trajectory.append(-token_lps.mean().item())

        # Cleanup
        del outputs, hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Scores
        third = max(1, n_layers // 3)
        early_nll = sum(nll_trajectory[:third]) / third
        late_nll = sum(nll_trajectory[-third:]) / third

        # LAC score: early-late discrepancy.
        # Positive → early layers more confident than late (unusual) → hallucination risk.
        # But more robust: use the trajectory variance.
        # Score = early_nll - late_nll  (positive = hallucination risk)
        score = early_nll - late_nll

        return LacResult(
            score=score,
            base_nll=base_nll,
            early_nll=early_nll,
            late_nll=late_nll,
            nll_trajectory=nll_trajectory,
            answer_token_count=n_ans,
        )

    @staticmethod
    def _compute_nll_from_logits(logits, input_ids, ans_start, ans_end) -> float:
        answer_logits = logits[:, ans_start - 1:ans_end - 1, :].float()
        answer_ids = input_ids[:, ans_start:ans_end].to(answer_logits.device)
        log_probs = F.log_softmax(answer_logits, dim=-1)
        token_lps = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
        return -token_lps.mean().item()
