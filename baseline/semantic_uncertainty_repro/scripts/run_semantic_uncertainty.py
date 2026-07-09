#!/usr/bin/env python3
"""Main entry point for semantic uncertainty evaluation.

Pipeline (three stages)
-----------------------
1. generate  – sample 1 low-temp + N high-temp answers per example
2. compute   – cluster responses, compute uncertainty scores
3. analyze   – aggregate metrics (AUROC, AURAC, …) → .summary.json

Usage
-----
  # Multimodal
  python scripts/run_semantic_uncertainty.py \
      --model llava --model-path /path/to/llava-1.5-7b-hf \
      --dataset cvbench --num-generations 5 --num-samples 200

  # Text-only
  python scripts/run_semantic_uncertainty.py \
      --model huggingface_llm --model-path meta-llama/Llama-2-7b-chat-hf \
      --dataset trivia_qa --num-generations 10 --num-samples 400

Environment
-----------
  SEM_UNC_MODELS_DIR   – root for model weights
  SEM_UNC_DATASETS_DIR – root for dataset cache
  VAUQ_MODELS_DIR / VAUQ_DATASETS_DIR – fallback (shared with vauq-repro)
  HF_HOME / HF_HUB_CACHE – Hugging Face cache
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Ensure the src directory is on the path.
_repo_root = Path(__file__).resolve().parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from sem_unc import build_dataset, build_entailment_model, build_model
from sem_unc.entailment import EntailmentModel
from sem_unc.metrics import compute_metrics as compute_eval_metrics
from sem_unc.models.base import Model
from sem_unc.semantic_entropy import compute_semantic_entropy
from sem_unc.types import SingleGeneration, SampleResult
from sem_unc.utils import get_metric_fn, setup_logger, write_json, write_jsonl

logger = logging.getLogger(__name__)

# Brief prompt prefix used by the original paper for text-only models.
BRIEF_PROMPT = "Answer the following question as briefly as possible.\n"


# ------------------------------------------------------------------
# Prompt helpers
# ------------------------------------------------------------------
def make_qa_prompt(question: str, answer: str | None = None) -> str:
    """Build the official short-phrase QA prompt without context."""
    prompt = f"Question: {question}\n"
    if answer is None:
        return prompt + "Answer:"
    return prompt + f"Answer: {answer}\n\n"


def build_fewshot_prompt(train_dataset, indices: list[int]) -> str:
    prompt = BRIEF_PROMPT
    for idx in indices:
        sample = train_dataset[idx]
        answer = sample["all_gt_ans"][0]
        prompt += make_qa_prompt(sample["question"], answer)
    return prompt


def official_triviaqa_indices(train_dataset, validation_dataset, args):
    """Match the official short-phrase RNG order for prompt/eval indices."""
    rng = random.Random(args.seed)
    answerable = [
        idx for idx in range(len(train_dataset))
        if train_dataset[idx].get("all_gt_ans")
    ]
    prompt_indices = rng.sample(answerable, args.num_few_shot)
    remaining_answerable = list(set(answerable) - set(prompt_indices))

    p_true_indices = rng.sample(answerable, 20)
    remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))

    train_possible = list(set(remaining_answerable))
    rng.sample(train_possible, min(args.num_samples, len(train_dataset)))

    validation_indices = rng.sample(
        range(len(validation_dataset)),
        min(args.num_samples, len(validation_dataset)),
    )
    return prompt_indices, validation_indices


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Semantic Uncertainty for Text + Multimodal Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- model ---
    p.add_argument(
        "--model", type=str, default="huggingface_llm",
        choices=["llava", "huggingface_llm"],
        help="Model registry key.",
    )
    p.add_argument(
        "--model-path", type=str, default=None,
        help="Path or HF repo id for the model.",
    )
    p.add_argument(
        "--load-in-8bit", action="store_true", default=False,
        help="Load text LLM in 8-bit (reduce VRAM).",
    )

    # --- dataset ---
    p.add_argument(
        "--dataset", type=str, default="trivia_qa",
        choices=["cvbench", "mmvet", "vilp", "trivia_qa", "squad", "nq", "svamp"],
        help="Dataset registry key.",
    )
    p.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to evaluate (default: all).",
    )

    # --- generation ---
    p.add_argument(
        "--num-generations", type=int, default=10,
        help="Number of high-temperature samples per question.",
    )
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature for high-t answers.",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Maximum tokens to generate per answer.",
    )
    p.add_argument(
        "--seed", type=int, default=10,
        help="Random seed for sampling.",
    )
    p.add_argument(
        "--num-few-shot", type=int, default=5,
        help="Number of official short-phrase few-shot examples.",
    )

    # --- entailment ---
    p.add_argument(
        "--entailment-model", type=str, default="deberta",
        help="Entailment model (default: deberta).",
    )
    p.add_argument(
        "--strict-entailment", action="store_true", default=False,
        help="Require strict bi-directional entailment for clustering.",
    )

    # --- accuracy metric ---
    p.add_argument(
        "--metric", type=str, default="squad",
        choices=["squad", "exact_match"],
        help="Metric to judge correctness.",
    )

    # --- output ---
    p.add_argument(
        "--output-dir", type=str, default="results/sem_unc",
        help="Output directory for JSONL and summary files.",
    )
    p.add_argument(
        "--tag", type=str, default=None,
        help="Optional tag for the output filename.",
    )

    # --- control ---
    p.add_argument(
        "--no-analyze", action="store_true", default=False,
        help="Skip the final analysis stage.",
    )
    p.add_argument(
        "--debug", action="store_true", default=False,
        help="Run in debug mode (fewer samples, more logging).",
    )

    return p.parse_args()


# ------------------------------------------------------------------
# Stage 1 – Generation
# ------------------------------------------------------------------
def run_generate(
    args: argparse.Namespace,
    model: Model,
    dataset,
    metric_fn: callable,
    output_path: Path,
    indices: list[int],
    fewshot_prompt: str,
) -> list[SampleResult]:
    """Generate 1 low-t + N high-t answers per sample."""
    logger.info("=" * 60)
    logger.info("STAGE 1 — GENERATE")
    logger.info("=" * 60)

    results: list[SampleResult] = []

    for idx in tqdm(indices, desc="generate"):
        sample = dataset[idx]
        raw_question = sample["question"]
        image = sample.get("img")
        gt_ans = sample["gt_ans"]
        all_gt_ans = sample.get("all_gt_ans", [gt_ans]) if gt_ans else []
        sample_id = sample["id"]

        model_input = fewshot_prompt + make_qa_prompt(raw_question)

        # --- 1 low-temperature answer ---
        ml_answer, ml_log_liks, ml_emb = model.generate(
            question=model_input,
            image=image,
            temperature=0.1,
            max_new_tokens=args.max_new_tokens,
        )

        is_correct = (
            _judge_correct(ml_answer, gt_ans, all_gt_ans, metric_fn)
            if gt_ans
            else False
        )

        most_likely = SingleGeneration(
            answer=ml_answer,
            token_log_likelihoods=ml_log_liks,
            embedding=ml_emb,
            accuracy=float(is_correct),
        )

        # --- N high-temperature answers ---
        high_temp: list[SingleGeneration] = []
        for _ in range(args.num_generations):
            ht_answer, ht_log_liks, ht_emb = model.generate(
                question=model_input,
                image=image,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            high_temp.append(
                SingleGeneration(
                    answer=ht_answer,
                    token_log_likelihoods=ht_log_liks,
                    embedding=ht_emb,
                )
            )

        result = SampleResult(
            id=sample_id,
            question=raw_question,
            gt_answers=list(all_gt_ans),
            most_likely_answer=most_likely,
            high_temp_answers=high_temp,
            correct=is_correct,
            prompt=model_input,
        )
        results.append(result)

        if args.debug and len(results) >= 5:
            break

        if (len(results) % 20) == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    records = [r.to_dict() for r in results]
    write_jsonl(output_path, records)
    logger.info("Saved %d generation results to %s", len(results), output_path)

    return results


def _judge_correct(
    prediction: str,
    gt_ans: str,
    all_gt_ans: list[str],
    metric_fn: callable,
) -> bool:
    """Check if prediction matches any ground-truth answer."""
    candidates = all_gt_ans if all_gt_ans else [gt_ans]
    for cand in candidates:
        if metric_fn(prediction, cand) >= 1.0:
            return True
    return False


# ------------------------------------------------------------------
# Stage 2 – Compute uncertainty
# ------------------------------------------------------------------
def run_compute(
    args: argparse.Namespace,
    results: list[SampleResult],
    entailment_model: EntailmentModel,
    output_path: Path,
) -> list[SampleResult]:
    """Compute semantic entropy and related scores for each sample."""
    logger.info("=" * 60)
    logger.info("STAGE 2 — COMPUTE UNCERTAINTY")
    logger.info("=" * 60)

    for result in tqdm(results, desc="compute"):
        responses = [ht.answer for ht in result.high_temp_answers]
        log_liks = [ht.token_log_likelihoods for ht in result.high_temp_answers]

        if len(responses) == 0:
            result.scores = {
                "semantic_entropy": 0.0,
                "regular_entropy": 0.0,
                "cluster_assignment_entropy": 0.0,
            }
            continue

        result.scores = compute_semantic_entropy(
            responses=responses,
            log_likelihoods=log_liks,
            entailment_model=entailment_model,
            strict_entailment=args.strict_entailment,
            question=result.question,
        )
        result.semantic_ids = result.scores.pop("semantic_ids")

    records = [r.to_dict() for r in results]
    write_jsonl(output_path, records)
    logger.info(
        "Saved %d results with uncertainty scores to %s",
        len(results),
        output_path,
    )

    return results


# ------------------------------------------------------------------
# Stage 3 – Analyze
# ------------------------------------------------------------------
def run_analyze(
    results: list[SampleResult],
    output_dir: Path,
    jsonl_path: Path,
) -> None:
    """Compute aggregate metrics and write a .summary.json file."""
    logger.info("=" * 60)
    logger.info("STAGE 3 — ANALYZE")
    logger.info("=" * 60)

    labels = [r.correct for r in results]
    sem_ent = [r.scores.get("semantic_entropy", 0.0) for r in results]
    reg_ent = [r.scores.get("regular_entropy", 0.0) for r in results]
    clust_ent = [
        r.scores.get("cluster_assignment_entropy", 0.0) for r in results
    ]

    summary = compute_eval_metrics(
        labels=labels,
        semantic_entropy=sem_ent,
        regular_entropy=reg_ent,
        cluster_entropy=clust_ent,
    )

    summary["n_samples"] = len(results)
    summary["n_correct"] = sum(labels)

    summary_path = jsonl_path.with_suffix(".summary.json")
    write_json(summary_path, summary)

    logger.info("--- Results ---")
    logger.info("Accuracy:    %.4f", summary["accuracy"])
    for name, m in summary["metrics"].items():
        logger.info(
            "%s  AUROC: %.4f  AURAC: %.4f",
            f"{name}:".ljust(32),
            m["auroc"],
            m["aurac"],
        )
    logger.info("Summary written to %s", summary_path)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    setup_logger(logging.DEBUG if args.debug else logging.INFO)

    logger.info("Semantic Uncertainty – run config: %s", args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.model}_{args.dataset}"
    jsonl_path = output_dir / f"{tag}.jsonl"

    # Resolve model path.
    model_path = args.model_path
    if model_path is None:
        model_path = os.environ.get("SEM_UNC_MODEL_PATH")
    if model_path is None:
        models_dir = os.environ.get("SEM_UNC_MODELS_DIR", "")
        if models_dir:
            model_path = os.path.join(models_dir, "Mistral-7B-Instruct-v0.1")
    if model_path is None:
        model_path = "mistralai/Mistral-7B-Instruct-v0.1"

    model = build_model(
        args.model,
        model_path=model_path,
        max_new_tokens=args.max_new_tokens,
        load_in_8bit=args.load_in_8bit,
    )
    train_dataset = build_dataset(args.dataset, split="train", seed=args.seed)
    dataset = build_dataset(args.dataset, split="validation", seed=args.seed)
    prompt_indices, eval_indices = official_triviaqa_indices(
        train_dataset, dataset, args
    )
    fewshot_prompt = build_fewshot_prompt(train_dataset, prompt_indices)
    logger.info("Few-shot prompt indices: %s", prompt_indices)
    logger.info("Evaluation sample count: %d", len(eval_indices))

    results = run_generate(
        args,
        model,
        dataset,
        get_metric_fn(args.metric),
        jsonl_path,
        indices=eval_indices,
        fewshot_prompt=fewshot_prompt,
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Stage 2: Compute uncertainty ---
    entailment_model = build_entailment_model(args.entailment_model)
    results = run_compute(args, results, entailment_model, jsonl_path)

    # --- Stage 3: Analyze ---
    if not args.no_analyze:
        run_analyze(results, output_dir, jsonl_path)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
