"""Visual and textual perturbations used by VL-Uncertainty."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

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
    visual_perturbation: str = "blurring"
    blur_radius_list: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    textual_perturbation: str = "llm_rephrasing"
    textual_temps: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    rephrase_template: str = DEFAULT_REPHRASE_TEMPLATE
    sampling_time: int = 5
    pair_order: str = "progressively"


def perturb_visual_prompt(image, config: PerturbationConfig) -> list:
    perturbed = []
    name = config.visual_perturbation
    if name == "none":
        return [image] * config.sampling_time
    if name == "blurring":
        for radius in config.blur_radius_list:
            perturbed.append(image_blurring(image, radius))
    elif name == "rotation":
        for degree in [-40, -20, 20, 40, 10]:
            perturbed.append(image_rotation(image, degree))
    elif name == "flipping":
        perturbed = [image_flipping(image, "horizontal")] * 2 + [image_flipping(image, "vertical")] * 3
    elif name == "shifting":
        for direction in ["up", "down", "left", "right"]:
            perturbed.append(image_shifting(image, direction, 100))
        perturbed.append(image_shifting(image, "up", 50))
    elif name == "cropping":
        for ratio in [0.95, 0.9, 0.85, 0.8, 0.75]:
            perturbed.append(image_cropping(image, ratio))
    elif name == "erasing":
        for size in [50, 60, 70, 80, 90, 100]:
            perturbed.append(image_erasing(image, erase_l=size, erase_w=size))
    elif name == "gaussian_noise":
        for degree in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed.append(gaussian_noise(image, degree))
    elif name == "dropout":
        for degree in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed.append(dropout(image, degree))
    elif name == "salt_and_pepper":
        for degree in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed.append(salt_and_pepper(image, degree))
    elif name == "sharpen":
        for degree in [0.1, 0.2, 0.3, 0.4, 0.5]:
            perturbed.append(image_sharpen(image, degree))
    elif name == "adjust_brightness":
        for degree in [0.8, 0.9, 1.1, 1.2, 1.3]:
            perturbed.append(adjust_brightness(image, degree))
    elif name == "adjust_contrast":
        for degree in [0.8, 0.9, 1.1, 1.2, 1.3]:
            perturbed.append(adjust_contrast(image, degree))
    elif name == "rotate_shift":
        for degree in [-40, -20, 20, 40, 10]:
            perturbed.append(image_shifting(image_rotation(image, degree), "up", 100))
    elif name == "crop_flip":
        for degree in [0.95, 0.9, 0.85, 0.8, 0.75]:
            perturbed.append(image_flipping(image_cropping(image, degree), "horizontal"))
    elif name == "rotate_blur":
        for degree in [-40, -20, 20, 40, 10]:
            perturbed.append(image_blurring(image_rotation(image, degree), 1))
    elif name == "crop_blur":
        for degree in [0.95, 0.9, 0.85, 0.8, 0.75]:
            perturbed.append(image_blurring(image_cropping(image, degree), 1))
    else:
        raise ValueError(f"Unknown visual perturbation {name!r}")
    return _fit_length(perturbed, config.sampling_time)


def perturb_textual_prompt(question: str, text_model: TextModel, config: PerturbationConfig) -> list[str]:
    original_question = parse_original_question(question)
    name = config.textual_perturbation
    perturbed: list[str] = []
    if name == "none":
        return [question] * config.sampling_time
    if name == "llm_rephrasing":
        for temp in config.textual_temps:
            instruction = config.rephrase_template.replace("{question}", original_question)
            perturbed_question = text_model.generate(instruction, temp, max_new_tokens=128)
            perturbed.append(merge_question(perturbed_question, question))
    elif name == "swapping":
        perturbed = [merge_question(word_swapping(original_question), question) for _ in range(config.sampling_time)]
    elif name == "deleting":
        perturbed = [merge_question(word_deleting(original_question), question) for _ in range(config.sampling_time)]
    elif name == "inserting":
        perturbed = [merge_question(word_inserting(original_question), question) for _ in range(config.sampling_time)]
    elif name == "replacing":
        perturbed = [merge_question(word_replacing(original_question), question) for _ in range(config.sampling_time)]
    elif name == "text_shuffle":
        perturbed = [merge_question(text_shuffle(original_question), question) for _ in range(config.sampling_time)]
    elif name == "noise_injection":
        for noise_level in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed.append(merge_question(noise_injection(original_question, noise_level), question))
    elif name == "word_dropout":
        for dropout_rate in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed.append(merge_question(word_dropout(original_question, dropout_rate), question))
    elif name == "character_dropout":
        for dropout_rate in [0.05, 0.1, 0.15, 0.2, 0.25]:
            perturbed.append(merge_question(character_dropout(original_question, dropout_rate), question))
    else:
        raise ValueError(f"Unknown textual perturbation {name!r}")
    return _fit_length(perturbed, config.sampling_time)


def combine_perturbed_prompts(sample: dict, images: list, questions: list[str], pair_order: str) -> list[dict]:
    questions = list(questions)
    if pair_order.startswith("shift"):
        shift_by = int(pair_order.split("_")[1])
        shift_by %= len(questions)
        questions = questions[shift_by:] + questions[:shift_by]
    elif pair_order == "random_pair":
        random.shuffle(questions)
    elif pair_order != "progressively":
        raise ValueError(f"Unknown pair_order {pair_order!r}")
    count = min(len(images), len(questions))
    prompts = []
    for i in range(count):
        item = sample.copy()
        item["img"] = images[i]
        item["question"] = questions[i]
        prompts.append(item)
    return prompts


def image_blurring(image, blur_radius):
    try:
        return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    except Exception:
        return image


def image_rotation(image, angle):
    try:
        return image.rotate(angle)
    except Exception:
        return image


def image_flipping(image, direction):
    try:
        if direction == "horizontal":
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        if direction == "vertical":
            return image.transpose(Image.FLIP_TOP_BOTTOM)
    except Exception:
        return image
    return image


def image_shifting(image, direction, length):
    try:
        w, h = image.size
        translation = {
            "up": (0, -length),
            "down": (0, length),
            "left": (-length, 0),
            "right": (length, 0),
        }[direction]
        return image.transform(
            (w, h),
            Image.AFFINE,
            (1, 0, translation[0], 0, 1, translation[1]),
            fillcolor=(0, 0, 0),
        )
    except Exception:
        return image


def image_cropping(image, scale=0.9):
    try:
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        return image.crop((left, top, left + new_w, top + new_h))
    except Exception:
        return image


def image_erasing(image, erase_l=50, erase_w=50):
    try:
        w, h = image.size
        erase_l = min(erase_l, h)
        erase_w = min(erase_w, w)
        top_left_x = random.randint(0, w - erase_w)
        top_left_y = random.randint(0, h - erase_l)
        erased_image = image.copy()
        erase_area = Image.new("RGB", (erase_w, erase_l), (0, 0, 0))
        erased_image.paste(erase_area, (top_left_x, top_left_y))
        return erased_image
    except Exception:
        return image


def adjust_brightness(image, factor):
    try:
        return ImageEnhance.Brightness(image).enhance(factor)
    except Exception:
        return image


def adjust_contrast(image, factor):
    try:
        return ImageEnhance.Contrast(image).enhance(factor)
    except Exception:
        return image


def gaussian_noise(image, degree):
    try:
        image_array = np.array(image)
        noise = np.random.normal(0, degree * 255, image_array.shape)
        return Image.fromarray(np.clip(image_array + noise, 0, 255).astype(np.uint8))
    except Exception:
        return image


def dropout(image, p):
    try:
        image_array = np.array(image)
        mask = np.expand_dims(np.random.rand(*image_array.shape[:2]) > p, axis=-1)
        return Image.fromarray((image_array * mask).astype(np.uint8))
    except Exception:
        return image


def salt_and_pepper(image, p):
    try:
        image_array = np.array(image, dtype=np.uint8)
        noise = np.random.rand(image_array.shape[0], image_array.shape[1])
        salt_mask = np.expand_dims(noise < (p / 2), axis=-1).repeat(3, axis=-1)
        pepper_mask = np.expand_dims(noise > 1 - (p / 2), axis=-1).repeat(3, axis=-1)
        image_array[salt_mask] = 255
        image_array[pepper_mask] = 0
        return Image.fromarray(image_array)
    except Exception:
        return image


def image_sharpen(image, degree):
    try:
        return ImageEnhance.Sharpness(image).enhance(degree)
    except Exception:
        return image


def word_swapping(question):
    try:
        words = question.split()
        if len(words) < 2:
            return question
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)
    except Exception:
        return question


def word_deleting(question):
    try:
        words = question.split()
        if len(words) <= 1:
            return question
        del words[random.randint(0, len(words) - 1)]
        return " ".join(words)
    except Exception:
        return question


def word_inserting(question):
    try:
        words = question.split()
        if not words:
            return question
        words.insert(random.randint(0, len(words)), random.choice(words))
        return " ".join(words)
    except Exception:
        return question


def word_replacing(question):
    try:
        words = question.split()
        if len(words) < 2:
            return question
        idx_to_replace = random.randint(0, len(words) - 1)
        candidates = [w for i, w in enumerate(words) if i != idx_to_replace]
        words[idx_to_replace] = random.choice(candidates)
        return " ".join(words)
    except Exception:
        return question


def text_shuffle(question):
    try:
        words = question.split()
        random.shuffle(words)
        return " ".join(words)
    except Exception:
        return question


def noise_injection(question, noise_level=0.1):
    try:
        chars = list(question)
        num_noisy_chars = int(noise_level * len(chars))
        for _ in range(num_noisy_chars):
            chars[random.randint(0, len(chars) - 1)] = random.choice(string.ascii_letters)
        return "".join(chars)
    except Exception:
        return question


def word_dropout(question, dropout_rate=0.1):
    try:
        return " ".join([word for word in question.split() if random.random() > dropout_rate])
    except Exception:
        return question


def character_dropout(question, dropout_rate=0.1):
    try:
        return "".join([char for char in question if random.random() > dropout_rate])
    except Exception:
        return question


def _fit_length(values: list, target: int) -> list:
    if not values:
        raise ValueError("Perturbation produced no values")
    if len(values) >= target:
        return values[:target]
    out = list(values)
    while len(out) < target:
        out.append(values[len(out) % len(values)])
    return out
