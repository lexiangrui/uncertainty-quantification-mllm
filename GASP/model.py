from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

from config import LLAVA_MODEL, MAX_NEW_TOKENS, REPLACEMENT_NOISE_SCALE
from perturb import perturb_embeddings


class LlavaRunner:
    def __init__(self, model_path: Path = LLAVA_MODEL):
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
        # LLaVA has already projected and merged visual vectors when this module
        # is called; its inputs_embeds have not entered the first self-attention.
        self.pre_attention_module = getattr(self.model, "language_model", None)
        if self.pre_attention_module is None:
            self.pre_attention_module = self.model.model.language_model
        self.image_token_id = int(self.model.config.image_token_index)

    def prepare_inputs(self, image, question: str) -> dict[str, torch.Tensor]:
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        return {name: tensor.to(self.device) for name, tensor in inputs.items()}

    @torch.inference_mode()
    def greedy_generate(self, inputs: dict[str, torch.Tensor]) -> dict:
        prompt_length = inputs["input_ids"].shape[1]
        sequences = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )
        answer_ids = sequences[:, prompt_length:]
        if answer_ids.numel() == 0:
            raise RuntimeError("LLaVA generated an empty answer")
        text = self.processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
        return {"text": text, "answer_ids": answer_ids.detach()}

    def _pre_attention_layout(
        self,
        full_ids: torch.Tensor,
        prompt_length: int,
        fused_length: int,
    ) -> tuple[torch.Tensor, int]:
        """Map visual positions and prompt length to the projector-merged sequence."""
        if full_ids.shape[0] != 1:
            raise ValueError("gradient tracing currently requires a batch of one")
        ids = full_ids[0]
        image_positions = torch.nonzero(ids == self.image_token_id, as_tuple=True)[0]
        if image_positions.numel() == 0:
            raise ValueError("the prompt contains no image token")

        visual_mask = torch.zeros(fused_length, dtype=torch.bool, device=ids.device)
        if fused_length == ids.numel():
            visual_mask = ids == self.image_token_id
            fused_prompt_length = prompt_length
        else:
            # Legacy processors emit one <image> placeholder. LLaVA expands it
            # to all projected image vectors before calling the language model.
            if image_positions.numel() != 1:
                raise ValueError("cannot map multiple legacy image placeholders")
            image_position = int(image_positions[0])
            visual_count = fused_length - ids.numel() + 1
            if visual_count <= 0:
                raise ValueError("invalid projector-merged sequence length")
            visual_mask[image_position : image_position + visual_count] = True
            fused_prompt_length = prompt_length + visual_count - 1

        if not visual_mask.any():
            raise ValueError("visual candidate set must be non-empty")
        return visual_mask, fused_prompt_length

    def score_fixed_answer(
        self,
        inputs: dict[str, torch.Tensor],
        answer_ids: torch.Tensor,
        *,
        capture_embedding_grad: bool = False,
        replacement_indices: torch.Tensor | None = None,
        reference_indices: torch.Tensor | None = None,
        replacement_seed: int | None = None,
        perturbation_mode: str = "replace",
        norm_isotropic_sigma: float = 0.01,
        return_last_hidden: bool = False,
    ) -> dict:
        if capture_embedding_grad and replacement_indices is not None:
            raise ValueError("gradient capture and Gaussian replacement are mutually exclusive")
        replacement_requested = replacement_indices is not None
        if replacement_requested != (reference_indices is not None) or replacement_requested != (
            replacement_seed is not None
        ):
            raise ValueError("replacement indices, reference indices, and seed must be supplied together")

        # generate() returns inference tensors; clone the target for autograd.
        answer_ids = answer_ids.clone()
        prompt_ids = inputs["input_ids"]
        prompt_length = prompt_ids.shape[1]
        full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        full_attention = torch.cat(
            [inputs["attention_mask"], torch.ones_like(answer_ids, device=self.device)], dim=1
        )
        trace: dict[str, torch.Tensor | int] = {}

        def pre_attention_hook(_module, args, kwargs):
            embeddings = kwargs.get("inputs_embeds")
            if embeddings is None:
                raise RuntimeError("LLaVA did not pass fused inputs_embeds to the language model")
            visual_mask, fused_prompt_length = self._pre_attention_layout(
                full_ids, prompt_length, embeddings.shape[1]
            )
            trace["visual_mask"] = visual_mask
            trace["fused_prompt_length"] = fused_prompt_length
            if capture_embedding_grad:
                leaf = embeddings.detach().requires_grad_(True)
                trace["embeddings"] = leaf
                kwargs["inputs_embeds"] = leaf
            elif replacement_requested:
                kwargs["inputs_embeds"] = perturb_embeddings(
                    embeddings,
                    replacement_indices,
                    reference_indices,
                    mode=perturbation_mode,
                    seed=replacement_seed,
                    replacement_scale=REPLACEMENT_NOISE_SCALE,
                    norm_isotropic_sigma=norm_isotropic_sigma,
                )
            return args, kwargs

        handle = self.pre_attention_module.register_forward_pre_hook(pre_attention_hook, with_kwargs=True)
        context = torch.enable_grad() if capture_embedding_grad else torch.inference_mode()
        try:
            with context:
                output = self.model(
                    input_ids=full_ids,
                    attention_mask=full_attention,
                    pixel_values=inputs["pixel_values"],
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=return_last_hidden,
                )
                response_start = int(trace["fused_prompt_length"]) - 1
                response_logits = output.logits[
                    :, response_start : response_start + answer_ids.shape[1], :
                ]
                if response_logits.shape[1] != answer_ids.shape[1]:
                    raise AssertionError("response logits and answer tokens are misaligned")
                nll = F.cross_entropy(
                    response_logits.float().reshape(-1, response_logits.shape[-1]),
                    answer_ids.reshape(-1),
                    reduction="mean",
                )
                result = {"mean_nll": float(nll.detach().item())}
                if return_last_hidden:
                    if output.hidden_states is None or not output.hidden_states:
                        raise RuntimeError("LLaVA did not return hidden states")
                    last_hidden = output.hidden_states[-1][0, -1, :].float()
                    if last_hidden.ndim != 1 or not torch.isfinite(last_hidden).all():
                        raise FloatingPointError("invalid last-token hidden vector")
                    result["last_hidden"] = last_hidden.detach().cpu()
                if capture_embedding_grad:
                    embeddings = trace["embeddings"]
                    grad = torch.autograd.grad(nll, embeddings, retain_graph=False, create_graph=False)[0]
                    scores = grad.float().norm(p=2, dim=-1)[0]
                    if not torch.isfinite(scores).all() or not torch.isfinite(nll):
                        raise FloatingPointError("non-finite pre-attention gradient or NLL")
                    result.update(
                        gradient_scores=scores.detach(),
                        visual_mask=trace["visual_mask"].detach(),
                    )
                return result
        finally:
            handle.remove()
