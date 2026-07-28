"""Visual and textual perturbations used by VL-Uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import ImageFilter

from vl_uncertainty.text_models import TextModel
from vl_uncertainty.utils import merge_question, parse_original_question


DEFAULT_REPHRASE_TEMPLATE = (
    "Given the input question: '{question}', generate a semantically equivalent "
    "variation by changing the wording, structure, grammar, or narrative. Ensure "
    "the perturbed question maintains the same meaning as the original. Provide "
    "only the rephrased question as the output."
)


@dataclass
class PerturbationConfig:
    blur_radius_list: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    textual_temps: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    rephrase_template: str = DEFAULT_REPHRASE_TEMPLATE


def perturb_visual_prompt(image, config: PerturbationConfig) -> list:
    image = image.convert("RGB")
    return [image.filter(ImageFilter.GaussianBlur(radius=radius)) for radius in config.blur_radius_list]


def perturb_textual_prompt(question: str, text_model: TextModel, config: PerturbationConfig) -> list[str]:
    original_question = parse_original_question(question)
    instruction = config.rephrase_template.replace("{question}", original_question)
    generated = text_model.generate_batch(
        [instruction] * len(config.textual_temps),
        list(config.textual_temps),
        max_new_tokens=256,
    )
    return [merge_question(item, question) for item in generated]


def combine_perturbed_prompts(sample: dict, images: list, questions: list[str]) -> list[dict]:
    prompts = []
    for image, question in zip(images, questions, strict=True):
        item = sample.copy()
        item["img"] = image
        item["question"] = question
        prompts.append(item)
    return prompts
