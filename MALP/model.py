import logging
from dataclasses import replace as dc_replace
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from config import LLAVA_MODEL, MAX_NEW_TOKENS, REASONING_LAYERS
from perturb import PerturbSpec, perturb_tensor


LOGGER = logging.getLogger("malp.model")


class LlavaMalpRunner:
    def __init__(self, model_path: Path = LLAVA_MODEL):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda:0",
            local_files_only=True,
            attn_implementation="sdpa",
        ).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = next(self.model.parameters()).device
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "left"
        self.projector = self.model.model.multi_modal_projector
        self.embedding_layer = self.model.get_input_embeddings()
        self.language_model = self.model.model.language_model
        self.decoder_layers = self.language_model.layers
        if max(REASONING_LAYERS) >= len(self.decoder_layers):
            raise RuntimeError(
                f"reasoning layer {max(REASONING_LAYERS)} is outside "
                f"the model's {len(self.decoder_layers)} decoder layers"
            )
        self.image_token_index = getattr(self.model.config, "image_token_index", None)
        if self.image_token_index is None:
            raise RuntimeError("model.config.image_token_index is required for text perturbation masking")
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            self.eos_token_ids = (eos_token_id,)
        elif eos_token_id is None:
            self.eos_token_ids = ()
        else:
            self.eos_token_ids = tuple(eos_token_id)

    @staticmethod
    def _model_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Drop MALP-only metadata before calling Hugging Face models."""
        return {name: value for name, value in inputs.items() if name != "question_token_mask"}

    def _build_question_token_mask(
        self,
        prompt: str,
        question: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Map the exact question character span in the rendered prompt to tokens."""
        start = prompt.find(question)
        if start < 0 or prompt.find(question, start + 1) >= 0:
            raise ValueError("question must occur exactly once in the rendered chat prompt")
        end = start + len(question)
        encoded = self.processor.tokenizer(
            prompt,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        tokenized_ids = encoded["input_ids"][0]
        offsets = encoded["offset_mapping"][0]
        tokenizer_mask = (offsets[:, 0] < end) & (offsets[:, 1] > start)
        tokenizer_mask &= offsets[:, 1].gt(offsets[:, 0])
        # Newer LLaVA processors expand one <image> placeholder into many image
        # tokens. Align all ordinary tokens and collapse/skip that expanded run.
        processor_ids = input_ids[0].cpu()
        processor_mask = torch.zeros_like(processor_ids, dtype=torch.bool)
        processor_index = 0
        for tokenizer_index, token_id in enumerate(tokenized_ids.tolist()):
            if token_id == self.image_token_index:
                if (
                    processor_index >= processor_ids.numel()
                    or processor_ids[processor_index].item() != self.image_token_index
                ):
                    raise RuntimeError("processor image-token expansion could not be aligned")
                while (
                    processor_index < processor_ids.numel()
                    and processor_ids[processor_index].item() == self.image_token_index
                ):
                    processor_index += 1
                continue
            if (
                processor_index >= processor_ids.numel()
                or processor_ids[processor_index].item() != token_id
            ):
                raise RuntimeError("processor and tokenizer input_ids could not be aligned")
            processor_mask[processor_index] = tokenizer_mask[tokenizer_index]
            processor_index += 1
        if processor_index != processor_ids.numel():
            raise RuntimeError("processor produced unaligned trailing input tokens")
        mask = processor_mask.unsqueeze(0).to(device=input_ids.device)
        mask &= attention_mask.bool()
        mask &= input_ids.ne(self.image_token_index)
        if not mask.any():
            raise RuntimeError("failed to map the question to any prompt tokens")
        return mask

    def prepare_inputs(self, image, question: str) -> dict[str, torch.Tensor]:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        inputs["question_token_mask"] = self._build_question_token_mask(
            prompt, question, inputs["input_ids"], inputs["attention_mask"]
        )
        return inputs

    def prepare_batch_inputs(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        prompts = []
        images = []
        for sample in samples:
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": sample["question"]}]}
            ]
            prompts.append(self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
            images.append(sample["image"])
        inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt")
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    @torch.inference_mode()
    def greedy_generate(self, inputs: dict[str, torch.Tensor]) -> dict:
        prompt_length = inputs["input_ids"].shape[1]
        sequences = self.model.generate(
            **self._model_inputs(inputs),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        answer_ids = sequences.sequences[:, prompt_length:]
        if answer_ids.numel() == 0:
            raise RuntimeError("LLaVA generated an empty answer")
        text = self.processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        answer_mask = self.build_answer_mask(answer_ids)
        return {"text": text, "answer_ids": answer_ids.detach(), "answer_mask": answer_mask.detach(), "scores": tuple(s.detach() for s in sequences.scores)}

    @torch.inference_mode()
    def greedy_generate_batch(self, inputs: dict[str, torch.Tensor]) -> dict:
        prompt_length = inputs["input_ids"].shape[1]
        sequences = self.model.generate(
            **self._model_inputs(inputs),
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )
        answer_ids = sequences[:, prompt_length:]
        if answer_ids.numel() == 0:
            raise RuntimeError("LLaVA generated an empty answer")
        texts = [text.strip() for text in self.processor.batch_decode(answer_ids, skip_special_tokens=True)]
        answer_mask = self.build_answer_mask(answer_ids)
        return {"texts": texts, "answer_ids": answer_ids.detach(), "answer_mask": answer_mask.detach()}

    def _register_generation_perturb_hooks(
        self,
        inputs: dict[str, torch.Tensor],
        spec: PerturbSpec,
    ) -> list:
        """Install perturbation hooks for cached autoregressive generation.

        Fusion representations are perturbed once during prompt prefill.  For
        reasoning perturbations, the final prefill position and every later
        decode position are perturbed, because those positions predict the
        generated answer tokens under causal decoding.  final_hidden is a
        generation-only intervention: it perturbs the last prompt-prefill
        state once and leaves all cached decode steps untouched.
        """
        if spec.modality != "joint":
            raise ValueError("fusion/reasoning perturbations require modality='joint'")
        if spec.mode == "adversarial":
            raise ValueError("answer-consistency generation does not support adversarial mode")

        handles = []
        if spec.stage == "fusion":
            prompt_mask = inputs["input_ids"].eq(self.image_token_index)
            question_mask = inputs.get("question_token_mask")
            if not torch.is_tensor(question_mask):
                raise TypeError("generation fusion perturbation requires question_token_mask")
            prompt_mask |= question_mask.bool()
            prefill_done = False

            def fusion_pre_hook(_module, args, kwargs):
                nonlocal prefill_done
                if prefill_done:
                    return args, kwargs
                if "inputs_embeds" not in kwargs:
                    raise RuntimeError("language_model did not receive inputs_embeds")
                value = kwargs["inputs_embeds"]
                if value.shape[:2] != prompt_mask.shape:
                    raise AssertionError(
                        "generation fusion mask does not align with prefill embeddings: "
                        f"mask={tuple(prompt_mask.shape)}, embeddings={tuple(value.shape[:2])}"
                    )
                kwargs["inputs_embeds"] = perturb_tensor(
                    value, replace(spec, token_mask=prompt_mask)
                )
                prefill_done = True
                return args, kwargs

            handles.append(
                self.language_model.register_forward_pre_hook(
                    fusion_pre_hook, with_kwargs=True
                )
            )
            return handles

        if spec.stage != "reasoning":
            if spec.stage == "final_hidden":
                applied = False
                def final_hook(_module, _args, output):
                    nonlocal applied
                    if applied:
                        return output
                    hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
                    mask = torch.zeros(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                    mask[:, -1] = True
                    local = replace(spec, token_mask=mask, seed=spec.seed)
                    applied = True
                    value = perturb_tensor(hidden, local)
                    if hasattr(output, "last_hidden_state"):
                        return dc_replace(output, last_hidden_state=value)
                    return (value, *output[1:])
                handles.append(self.language_model.register_forward_hook(final_hook))
                return handles
            raise ValueError(f"unknown perturbation stage: {spec.stage}")

        # Each layer is called once per model forward.  A per-layer counter
        # gives every decode step independent, deterministic noise without
        # coupling the counters of the six hooked layers.
        layer_steps = {layer_index: 0 for layer_index in REASONING_LAYERS}

        def attention_hook(layer_index, output):
            value = output[0] if isinstance(output, tuple) else output
            step = layer_steps[layer_index]
            layer_steps[layer_index] = step + 1
            token_mask = torch.zeros(value.shape[:2], dtype=torch.bool, device=value.device)
            if step == 0:
                # Single-sample generation has no right padding.  The final
                # prefill hidden state predicts the first generated token.
                token_mask[:, -1] = True
            else:
                # With a KV cache this is normally one position; selecting all
                # positions also remains correct for cache implementations
                # that submit a short block of uncached tokens.
                token_mask[:] = True
            layer_spec = replace(
                spec,
                token_mask=token_mask,
                seed=spec.seed + layer_index * 100_000 + step,
            )
            perturbed = perturb_tensor(value, layer_spec)
            return (perturbed, *output[1:]) if isinstance(output, tuple) else perturbed

        for layer_index in REASONING_LAYERS:
            handles.append(
                self.decoder_layers[layer_index].self_attn.register_forward_hook(
                    lambda _module, _args, output, index=layer_index: attention_hook(
                        index, output
                    )
                )
            )
        return handles

    @torch.inference_mode()
    def generate_with_perturbation(
        self,
        inputs: dict[str, torch.Tensor],
        spec: PerturbSpec,
    ) -> dict:
        """Greedily regenerate an answer while latent perturbations are active."""
        prompt_length = inputs["input_ids"].shape[1]
        handles = self._register_generation_perturb_hooks(inputs, spec)
        try:
            sequences = self.model.generate(
                **self._model_inputs(inputs),
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        answer_ids = sequences.sequences[:, prompt_length:]
        if answer_ids.numel() == 0:
            raise RuntimeError("LLaVA generated an empty perturbed answer")
        text = self.processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        return {"text": text, "answer_ids": answer_ids.detach(), "scores": tuple(s.detach() for s in sequences.scores)}

    def build_answer_mask(self, answer_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(answer_ids, dtype=torch.bool, device=answer_ids.device)
        if not self.eos_token_ids:
            return mask
        is_eos = torch.zeros_like(mask)
        for token_id in self.eos_token_ids:
            is_eos |= answer_ids.eq(token_id)
        # Keep the first EOS: predicting when the answer terminates is part of
        # its likelihood. Later EOS tokens are generate() batch padding.
        after_first_eos = (is_eos.cumsum(dim=1) - is_eos.to(torch.int64)).gt(0)
        mask &= ~after_first_eos
        if self.pad_token_id is not None and self.pad_token_id not in self.eos_token_ids:
            mask &= answer_ids.ne(self.pad_token_id)
        return mask

    def build_teacher_forcing_inputs(
        self,
        inputs: dict[str, torch.Tensor],
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int]:
        answer_ids = answer_ids.to(self.device).clone()
        if answer_mask is None:
            answer_mask = torch.ones_like(answer_ids, dtype=torch.bool, device=self.device)
        else:
            answer_mask = answer_mask.to(self.device).bool().clone()
        prompt_ids = inputs["input_ids"]
        prompt_length = prompt_ids.shape[1]
        full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        full_attention = torch.cat(
            [inputs["attention_mask"], answer_mask.to(dtype=inputs["attention_mask"].dtype)],
            dim=1,
        )
        return {
            "input_ids": full_ids,
            "attention_mask": full_attention,
            "pixel_values": inputs["pixel_values"],
            "question_token_mask": inputs["question_token_mask"],
            "prompt_length": prompt_length,
            "answer_length": answer_ids.shape[1],
            "answer_ids": answer_ids,
            "answer_mask": answer_mask,
        }

    def build_text_prompt_mask(self, teacher_inputs: dict[str, torch.Tensor | int]) -> torch.Tensor:
        question_mask = teacher_inputs.get("question_token_mask")
        prompt_length = int(teacher_inputs["prompt_length"])
        answer_length = int(teacher_inputs["answer_length"])
        if not torch.is_tensor(question_mask):
            raise TypeError("teacher inputs must contain a tensor question_token_mask")
        false_answer = torch.zeros(
            question_mask.shape[0], answer_length, dtype=torch.bool, device=question_mask.device
        )
        mask = torch.cat([question_mask.bool(), false_answer], dim=1)
        if mask.shape[1] != prompt_length + answer_length:
            raise AssertionError("question token mask and teacher-forcing inputs are misaligned")
        return mask

    def build_fusion_mask(self, teacher_inputs: dict[str, torch.Tensor | int]) -> torch.Tensor:
        """Select fused image tokens plus exact question tokens, excluding answer/template tokens."""
        input_ids = teacher_inputs["input_ids"]
        question_mask = teacher_inputs.get("question_token_mask")
        prompt_length = int(teacher_inputs["prompt_length"])
        answer_length = int(teacher_inputs["answer_length"])
        if not torch.is_tensor(input_ids) or not torch.is_tensor(question_mask):
            raise TypeError("fusion mask requires tensor input_ids and question_token_mask")
        image_prompt = input_ids[:, :prompt_length].eq(self.image_token_index)
        prompt_mask = image_prompt | question_mask.bool()
        answer_zeros = torch.zeros(
            prompt_mask.shape[0], answer_length, dtype=torch.bool, device=prompt_mask.device
        )
        mask = torch.cat([prompt_mask, answer_zeros], dim=1)
        if mask.shape != input_ids.shape or not mask.any():
            raise AssertionError("fusion mask is empty or misaligned")
        return mask

    @staticmethod
    def build_reasoning_mask(
        teacher_inputs: dict[str, torch.Tensor | int],
    ) -> torch.Tensor:
        """Select causal hidden positions that predict the fixed answer tokens."""
        attention_mask = teacher_inputs["attention_mask"]
        prompt_length = int(teacher_inputs["prompt_length"])
        answer_length = int(teacher_inputs["answer_length"])
        answer_mask = teacher_inputs["answer_mask"]
        if not torch.is_tensor(attention_mask) or not torch.is_tensor(answer_mask):
            raise TypeError("reasoning mask requires tensor attention_mask and answer_mask")
        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        mask[:, prompt_length - 1 : prompt_length + answer_length - 1] = answer_mask.bool()
        return mask

    def build_input_text_mask(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.bool().clone()
        mask &= input_ids.ne(self.image_token_index)
        if self.pad_token_id is not None:
            mask &= input_ids.ne(self.pad_token_id)
        return mask

    @staticmethod
    def response_logits(logits: torch.Tensor, prompt_length: int, answer_length: int) -> torch.Tensor:
        del prompt_length
        if logits.shape[1] < answer_length + 1:
            raise AssertionError("not enough logits to score the answer tokens")
        # The answer is appended at the tail. This remains correct whether
        # image tokens are expanded by the processor or inside model.forward().
        response = logits[:, -(answer_length + 1) : -1, :]
        if response.shape[1] != answer_length:
            raise AssertionError("response logits and answer tokens are misaligned")
        return response

    @staticmethod
    def per_sample_nll(
        response_logits: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if response_logits.shape[:2] != answer_ids.shape:
            raise AssertionError(
                f"response logits shape {tuple(response_logits.shape)} does not align with "
                f"answer ids shape {tuple(answer_ids.shape)}"
            )
        losses = F.cross_entropy(
            response_logits.float().reshape(-1, response_logits.shape[-1]),
            answer_ids.reshape(-1),
            reduction="none",
        ).reshape(answer_ids.shape)
        if answer_mask is None:
            answer_mask = torch.ones_like(answer_ids, dtype=torch.bool, device=answer_ids.device)
        mask = answer_mask.to(device=losses.device, dtype=losses.dtype)
        return (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def mean_nll(
        response_logits: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor | None = None,
    ) -> float:
        loss = LlavaMalpRunner.per_sample_nll(response_logits, answer_ids, answer_mask).mean()
        return float(loss.detach().item())

    @staticmethod
    def total_logprob(
        response_logits: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if response_logits.shape[:2] != answer_ids.shape:
            raise AssertionError(
                f"response logits shape {tuple(response_logits.shape)} does not align with "
                f"answer ids shape {tuple(answer_ids.shape)}"
            )
        log_probs = F.log_softmax(response_logits.float(), dim=-1)
        selected = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
        if answer_mask is None:
            return selected.sum()
        return (selected * answer_mask.to(device=selected.device, dtype=selected.dtype)).sum()

    @staticmethod
    def kl_divergence_per_sample(
        original_logits: torch.Tensor,
        perturbed_logits: torch.Tensor,
        answer_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if original_logits.shape != perturbed_logits.shape:
            raise AssertionError(
                f"original logits shape {tuple(original_logits.shape)} does not match "
                f"perturbed logits shape {tuple(perturbed_logits.shape)}"
            )
        original_log_probs = F.log_softmax(original_logits.float(), dim=-1)
        perturbed_log_probs = F.log_softmax(perturbed_logits.float(), dim=-1)
        original_probs = original_log_probs.exp()
        kl_by_token = (original_probs * (original_log_probs - perturbed_log_probs)).sum(dim=-1)
        if answer_mask is None:
            return kl_by_token.mean(dim=1)
        mask = answer_mask.to(device=kl_by_token.device, dtype=kl_by_token.dtype)
        return (kl_by_token * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def mean_kl(
        original_logits: torch.Tensor,
        perturbed_logits: torch.Tensor,
        answer_mask: torch.Tensor | None = None,
    ) -> float:
        kl = LlavaMalpRunner.kl_divergence_per_sample(
            original_logits, perturbed_logits, answer_mask
        ).mean()
        return float(kl.detach().item())

    @torch.inference_mode()
    def forward_original(self, teacher_inputs: dict[str, torch.Tensor | int]) -> dict:
        outputs = self.model(
            input_ids=teacher_inputs["input_ids"],
            attention_mask=teacher_inputs["attention_mask"],
            pixel_values=teacher_inputs["pixel_values"],
            use_cache=False,
            return_dict=True,
        )
        response = self.response_logits(
            outputs.logits,
            int(teacher_inputs["prompt_length"]),
            int(teacher_inputs["answer_length"]),
        )
        return {"response_logits": response.detach()}

    def _stage_module_and_mask(
        self,
        teacher_inputs: dict[str, torch.Tensor | int],
        stage: str,
    ) -> tuple[torch.nn.Module, torch.Tensor]:
        if stage == "fusion":
            return self.language_model, self.build_fusion_mask(teacher_inputs)
        if stage == "reasoning":
            return self.decoder_layers[REASONING_LAYERS[0]].self_attn, self.build_reasoning_mask(
                teacher_inputs
            )
        raise ValueError(f"unknown perturbation stage: {stage}")

    @torch.inference_mode()
    def forward_with_perturbation(
        self,
        teacher_inputs: dict[str, torch.Tensor | int],
        spec: PerturbSpec,
    ) -> dict:
        if spec.modality != "joint":
            raise ValueError("fusion/reasoning perturbations require modality='joint'")
        if spec.stage == "final_hidden":
            raise ValueError(
                "final_hidden is generation-only: perturb prompt-prefill once, then score "
                "generated answers with an unperturbed teacher-forcing forward"
            )
        handles = []
        if spec.stage == "fusion":
            token_mask = self.build_fusion_mask(teacher_inputs)
        elif spec.stage == "reasoning":
            token_mask = self.build_reasoning_mask(teacher_inputs)
        else:
            raise ValueError(f"unknown perturbation stage: {spec.stage}")
        masked_spec = replace(spec, token_mask=token_mask)

        def fusion_pre_hook(_module, args, kwargs):
            if "inputs_embeds" not in kwargs:
                raise RuntimeError("language_model did not receive inputs_embeds")
            kwargs["inputs_embeds"] = perturb_tensor(kwargs["inputs_embeds"], masked_spec)
            return args, kwargs

        def attention_hook(layer_index, output):
            if isinstance(output, tuple):
                value = output[0]
            else:
                value = output
            layer_spec = replace(masked_spec, seed=spec.seed + layer_index)
            if spec.adv_gradient is not None:
                gradient_index = REASONING_LAYERS.index(layer_index)
                layer_spec = replace(
                    layer_spec, adv_gradient=spec.adv_gradient[gradient_index]
                )
            perturbed = perturb_tensor(value, layer_spec)
            return (perturbed, *output[1:]) if isinstance(output, tuple) else perturbed

        try:
            if spec.stage == "fusion":
                handles.append(
                    self.language_model.register_forward_pre_hook(
                        fusion_pre_hook, with_kwargs=True
                    )
                )
            elif spec.stage == "reasoning":
                for layer_index in REASONING_LAYERS:
                    handles.append(
                        self.decoder_layers[layer_index].self_attn.register_forward_hook(
                            lambda _module, _inputs, output, index=layer_index: attention_hook(
                                index, output
                            )
                        )
                    )
            else:
                raise ValueError(f"unknown perturbation stage: {spec.stage}")
            outputs = self.model(
                input_ids=teacher_inputs["input_ids"],
                attention_mask=teacher_inputs["attention_mask"],
                pixel_values=teacher_inputs["pixel_values"],
                use_cache=False,
                return_dict=True,
            )
            response = self.response_logits(
                outputs.logits,
                int(teacher_inputs["prompt_length"]),
                int(teacher_inputs["answer_length"]),
            )
            return {"response_logits": response.detach()}
        finally:
            for handle in handles:
                handle.remove()

    def compute_logprob_gradient(
        self,
        teacher_inputs: dict[str, torch.Tensor | int],
        stage: str,
    ) -> dict[str, torch.Tensor]:
        handles = []
        trace: dict[int | str, torch.Tensor] = {}

        def capture_fusion(_module, args, kwargs):
            tracked = kwargs["inputs_embeds"].detach().requires_grad_(True)
            trace["embedding"] = tracked
            kwargs["inputs_embeds"] = tracked
            return args, kwargs

        def capture_reasoning(layer_index, output):
            value = output[0] if isinstance(output, tuple) else output
            tracked = value.detach().requires_grad_(True)
            trace[layer_index] = tracked
            return (tracked, *output[1:]) if isinstance(output, tuple) else tracked

        self.model.zero_grad(set_to_none=True)
        try:
            if stage == "fusion":
                handles.append(
                    self.language_model.register_forward_pre_hook(capture_fusion, with_kwargs=True)
                )
            elif stage == "reasoning":
                for layer_index in REASONING_LAYERS:
                    handles.append(
                        self.decoder_layers[layer_index].self_attn.register_forward_hook(
                            lambda _module, _inputs, output, index=layer_index: capture_reasoning(
                                index, output
                            )
                        )
                    )
            else:
                raise ValueError(f"unknown perturbation stage: {stage}")
            with torch.enable_grad():
                outputs = self.model(
                    input_ids=teacher_inputs["input_ids"],
                    attention_mask=teacher_inputs["attention_mask"],
                    pixel_values=teacher_inputs["pixel_values"],
                    use_cache=False,
                    return_dict=True,
                )
                response = self.response_logits(
                    outputs.logits,
                    int(teacher_inputs["prompt_length"]),
                    int(teacher_inputs["answer_length"]),
                )
                answer_ids = teacher_inputs["answer_ids"]
                answer_mask = teacher_inputs["answer_mask"]
                if not torch.is_tensor(answer_ids) or not torch.is_tensor(answer_mask):
                    raise TypeError("answer ids and mask must be tensors")
                self.total_logprob(response, answer_ids, answer_mask).backward()
            if stage == "fusion":
                embedding = trace.get("embedding")
                if embedding is None or embedding.grad is None:
                    raise RuntimeError("failed to capture fusion gradient")
                gradient = embedding.grad.detach()
            else:
                missing = [
                    index
                    for index in REASONING_LAYERS
                    if index not in trace or trace[index].grad is None
                ]
                if missing:
                    raise RuntimeError(f"failed to capture reasoning gradients at layers {missing}")
                gradient = torch.stack(
                    [trace[index].grad.detach() for index in REASONING_LAYERS], dim=0
                )
            non_finite = ~torch.isfinite(gradient)
            if non_finite.any():
                LOGGER.warning(
                    "%s gradient has %d/%d non-finite elements; sanitizing",
                    stage,
                    int(non_finite.sum().item()),
                    gradient.numel(),
                )
                gradient = torch.nan_to_num(gradient)
            return {"gradient": gradient, "response_logits": response.detach()}
        finally:
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)
