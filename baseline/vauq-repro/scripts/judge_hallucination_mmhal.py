#!/usr/bin/env python3
"""Multimodal MMHal-style judge for VAUQ experiment 2.

The judge directly observes the image and emits one correctness boolean plus
the official MMHal-Bench 0--6 rating. Hallucination is derived deterministically
as ``rating < 3``; it is never independently requested from the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vauq.benchmarks import build_benchmark


MMHAL_SYSTEM_PROMPT = r"""Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

The standard human-generated answer may only contain factual information and may not be comprehensive, so be cautious during evaluation. Detailed valid analysis or reasoning should be encouraged. Use these official MMHal-Bench ratings:
- Rating 6: very informative with good analysis or reasoning, no hallucination
- Rating 5: very informative, no hallucination
- Rating 4: somewhat informative, no hallucination
- Rating 3: not informative, no hallucination
- Rating 2: very informative, with hallucination
- Rating 1: somewhat informative, with hallucination
- Rating 0: not informative, with hallucination

Also judge correctness independently from hallucination. A response can be incorrect without hallucination, or correct while containing an additional hallucinated visual claim. Return exactly one JSON object with exactly these fields:
{"analysis":"brief justification","correct":true,"rating":4}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["mmvet", "vilp"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--model-path", default=os.environ.get("QWEN_MMHAL_MODEL"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    return parser.parse_args()


def load_model(model_path: str):
    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError as error:
        raise RuntimeError(
            "Qwen3.6 requires a Transformers release providing "
            "AutoModelForMultimodalLM; upgrade the judge environment."
        ) from error
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    return processor, model


def build_messages(dataset: str, image, question: str, reference: str, prediction: str) -> list[dict]:
    user_text = (
        f"[Dataset]\n{dataset}\n\n"
        f"[Question]\n{question}\n\n"
        f"[Accepted Reference Answer]\n{reference}\n\n"
        f"[Candidate Response]\n{prediction}\n\n"
        f"[Output]\n"
        f"Judge whether the response correctly answers the question, and assign the "
        f"official MMHal-Bench rating from 0 to 6. Return JSON only."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": MMHAL_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def parse_response(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {text[:300]!r}") from error
    if not isinstance(value, dict) or set(value) != {"analysis", "correct", "rating"}:
        raise ValueError(f"invalid fields: {text[:300]!r}")
    if not isinstance(value["analysis"], str) or not value["analysis"].strip():
        raise ValueError("analysis must be a non-empty string")
    if type(value["correct"]) is not bool:
        raise ValueError("correct must be a JSON boolean")
    if type(value["rating"]) is not int or not 0 <= value["rating"] <= 6:
        raise ValueError("rating must be an integer from 0 through 6")
    value["hallucination"] = value["rating"] < 3
    return value


@torch.inference_mode()
def judge_one(processor, model, dataset, image, question, reference, prediction, max_new_tokens):
    messages = build_messages(dataset, image, question, reference, prediction)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    first_device = next(model.parameters()).device
    inputs = inputs.to(first_device)
    output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    response = processor.decode(
        output[0, inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    return response, parse_response(response)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_done(path: Path) -> set[str]:
    return {str(row["id"]) for row in read_jsonl(path)} if path.exists() else set()


def score_metrics(rows: list[dict], label_key: str, condition=None) -> dict[str, float | int]:
    selected = [row for row in rows if condition is None or condition(row)]
    labels = np.asarray([int(row[label_key]) for row in selected])
    scores = np.asarray([float(row["scores"]["vauq"]) for row in selected])
    result: dict[str, float | int] = {"n": len(selected), "positive": int(labels.sum())}
    if len(selected) == 0 or len(set(labels.tolist())) < 2:
        result.update({"auroc": float("nan"), "aupr": float("nan")})
        return result
    result["auroc"] = float(roc_auc_score(labels, scores))
    result["aupr"] = float(average_precision_score(labels, scores))
    return result


def summarize(rows: list[dict]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("judge_status") == "ok"]
    quadrants = {key: 0 for key in ("correct_no_hall", "correct_hall", "error_no_hall", "error_hall")}
    for row in valid:
        key = (
            ("correct" if row["correct"] else "error") +
            ("_hall" if row["hallucination"] else "_no_hall")
        )
        quadrants[key] += 1
    ratings = [int(row["rating"]) for row in valid]
    return {
        "n": len(rows),
        "n_valid": len(valid),
        "n_failed": len(rows) - len(valid),
        "prompt_sha256": hashlib.sha256(MMHAL_SYSTEM_PROMPT.encode()).hexdigest(),
        "mean_rating": float(np.mean(ratings)) if ratings else float("nan"),
        "hallucination_rate": float(np.mean([row["hallucination"] for row in valid])) if valid else float("nan"),
        "quadrants": quadrants,
        "metrics": {
            "error_all": score_metrics(valid, "error"),
            "hallucination_all": score_metrics(valid, "hallucination"),
            "hallucination_given_error": score_metrics(valid, "hallucination", lambda row: row["error"]),
            "hallucination_given_correct": score_metrics(valid, "hallucination", lambda row: row["correct"]),
        },
    }


def main() -> None:
    args = parse_args()
    if not args.model_path:
        raise SystemExit("--model-path or QWEN_MMHAL_MODEL is required")
    input_rows = read_jsonl(Path(args.input))
    benchmark = build_benchmark(args.benchmark, question_style="raw")
    if len(input_rows) != benchmark.obtain_size():
        raise ValueError(
            f"input contains {len(input_rows)} rows but {args.benchmark} has {benchmark.obtain_size()} samples"
        )
    processor, model = load_model(args.model_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output_path) if args.resume else set()
    mode = "a" if args.resume and done else "w"
    limit = len(input_rows) if not args.limit or args.limit <= 0 else min(len(input_rows), args.start + args.limit)

    with output_path.open(mode, encoding="utf-8") as handle:
        for idx in tqdm(range(args.start, limit), desc=f"MMHal judge {args.benchmark}"):
            source = input_rows[idx]
            if str(source["id"]) in done:
                continue
            sample = benchmark.retrieve(idx)
            if str(source["id"]) != str(idx):
                raise ValueError(f"row order/id mismatch at {idx}")
            if source["question"] != sample["question"]:
                raise ValueError(f"question mismatch at {idx}")
            row = dict(source)
            try:
                raw, result = judge_one(
                    processor, model, args.benchmark, sample["img"], sample["question"],
                    str(sample.get("gt_ans")), source["prediction"], args.max_new_tokens,
                )
                row.update(result)
                row["error"] = not result["correct"]
                row["judge_status"] = "ok"
                row["judge_failure"] = None
                row["judge_raw_response"] = raw
            except ValueError as error:
                row.update({
                    "correct": None, "error": None, "rating": None,
                    "hallucination": None, "judge_status": "parse_failure",
                    "judge_failure": str(error),
                })
            row["judge"] = {
                "model_path": args.model_path,
                "protocol": "mmhal_official_rating_plus_independent_correctness_v1",
                "prompt_sha256": hashlib.sha256(MMHAL_SYSTEM_PROMPT.encode()).hexdigest(),
                "decoding": "greedy",
                "enable_thinking": False,
                "max_new_tokens": args.max_new_tokens,
                "hallucination_rule": "rating < 3",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    rows = read_jsonl(output_path)
    summary = summarize(rows)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
