#!/usr/bin/env python3
"""Run VL-Uncertainty evaluation (official algorithm, Zhang et al. 2024).

Pipeline
--------
1. For each sample, generate 1 low-temperature (t=0.1) answer.
2. Apply 5 visual perturbations (Gaussian blur).
3. Apply 5 textual perturbations (LLM rephrasing).
4. Pair perturbed prompts progressively, sample 1 answer each.
5. Cluster answers by semantic equivalence (LLM entailment for free-form).
6. Compute count-based Shannon entropy over clusters.
7. Evaluate hallucination detection accuracy, AUROC, AUPR.

Usage
-----
  # Multimodal — MM-Vet + LLaVA-1.5-7B
  python scripts/run_vl_uncertainty.py \\
      --backend llava --model-path /path/to/llava-1.5-7b-hf \\
      --benchmark mmvet --text-model qwen \\
      --text-model-path /path/to/Qwen2.5-3B-Instruct \\
      --limit 218 --output results/mmvet_llava_vlu.jsonl

  # Smoke test (no GPU / no models)
  python scripts/run_vl_uncertainty.py \\
      --backend mock --benchmark toy --text-model echo --judge choice --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vl_uncertainty.backends import BACKEND_MAP, Backend, build_backend
from vl_uncertainty.benchmarks import BENCHMARK_MAP, DEFAULT_JUDGE, Benchmark, build_benchmark
from vl_uncertainty.eval import compute_metrics
from vl_uncertainty.judges import build_judge
from vl_uncertainty.perturbations import PerturbationConfig
from vl_uncertainty.scoring import compute_vl_uncertainty
from vl_uncertainty.text_models import build_text_model


# ------------------------------------------------------------------
# Mock & Toy (smoke tests)
# ------------------------------------------------------------------
class MockBackend(Backend):
    def generate(self, image, question: str, temp: float = 0.1, max_new_tokens: int = 64):
        import torch
        if temp < 0.5:
            return "0", [-0.5, -0.3], None
        ans = str(int(temp * 10) % 2)
        return ans, [-0.2, -0.1], None


class ToyBenchmark(Benchmark):
    benchmark_type = "multi_choice"

    def __init__(self):
        from PIL import Image
        self.img = Image.new("RGB", (8, 8), (255, 255, 255))

    def obtain_size(self) -> int:
        return 2

    def retrieve(self, idx: int) -> dict | None:
        return {
            "idx": idx, "img": self.img, "num_c": 2,
            "question": "What color is the image?\n(0): white\n(1): black\nThis is a single choice question, answer only with choice number in 0, 1.",
            "gt_ans": idx % 2, "choices": ["white", "black"],
        }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="VL-Uncertainty (Zhang et al. 2024)")
    p.add_argument("--backend", choices=sorted([*BACKEND_MAP, "mock"]), default="llava")
    p.add_argument("--benchmark", choices=sorted([*BENCHMARK_MAP, "toy"]), default="mmvet")
    p.add_argument("--judge", choices=["choice", "llm", "none"], default=None)
    p.add_argument("--text-model", choices=["qwen", "echo"], default="echo")
    p.add_argument("--model-path", default=os.environ.get("VLU_MODEL_PATH", "llava-hf/llava-1.5-7b-hf"))
    p.add_argument("--text-model-path", default=os.environ.get("VLU_TEXT_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct"))
    p.add_argument("--llava-device", default="cuda:0")
    p.add_argument("--text-device", default="cuda:1")
    p.add_argument("--sampling-temp", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--perturbation-type", choices=["blurring", "none"], default="blurring")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--output", default="results/vl_uncertainty_results.jsonl")
    p.add_argument("--summary-output", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    _load_default_env_file()
    args = parse_args()
    fix_seed(args.seed)

    # Backend (LVLM on GPU 0).
    backend = MockBackend() if args.backend == "mock" else build_backend(
        args.backend, model_path=args.model_path, device=args.llava_device,
    )

    # Benchmark.
    benchmark = ToyBenchmark() if args.benchmark == "toy" else build_benchmark(args.benchmark)

    # Text model (Qwen for rephrasing, entailment, judge — on GPU 1).
    text_kwargs = {} if args.text_model == "echo" else {
        "model_path": args.text_model_path,
        "device": args.text_device,
    }
    text_model = build_text_model(args.text_model, **text_kwargs)

    # Judge.
    judge_name = args.judge or DEFAULT_JUDGE.get(args.benchmark, "none")
    judge = build_judge(judge_name, text_model=text_model)

    # Perturbation config.
    pert_config = PerturbationConfig(
        visual_perturbation=args.perturbation_type,
        blur_radius_list=(0.6, 0.8, 1.0, 1.2, 1.4),
        textual_perturbation="llm_rephrasing" if args.text_model != "echo" else "none",
        textual_temps=(0.1, 0.2, 0.3, 0.4, 0.5),
        sampling_time=5,
        pair_order="progressively",
    )

    # Output.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = _load_done(out_path) if args.resume else set()
    mode = "a" if args.resume and done_ids else "w"

    benchmark_type = getattr(benchmark, "benchmark_type", "free_form")
    size = benchmark.obtain_size()
    end = size if args.limit is None else min(size, args.start + args.limit)
    records: list[dict] = []

    with out_path.open(mode, encoding="utf-8") as f:
        for idx in tqdm(range(args.start, end), desc=f"vl-uncertainty {args.benchmark}"):
            idx_str = str(idx)
            if idx_str in done_ids:
                continue

            sample = benchmark.retrieve(idx)
            if sample is None:
                continue

            image = sample.get("img")
            raw_question = sample["question"]
            gt_ans = sample.get("gt_ans")

            # Compute VL-Uncertainty.
            result = compute_vl_uncertainty(
                backend=backend,
                sample={"img": image, "question": raw_question},
                text_model=text_model,
                benchmark_type=benchmark_type,
                pert_config=pert_config,
                sampling_temp=args.sampling_temp,
                max_new_tokens=args.max_new_tokens,
            )

            prediction = result.most_likely_answer or result.sampled_answers[0]

            # Judge correctness.
            correct = judge.judge(prediction, gt_ans, sample)

            hallucination_pred = result.uncertainty >= 1.0
            detection_correct = None if correct is None else (
                (correct and not hallucination_pred) or (not correct and hallucination_pred)
            )

            row = {
                "id": idx_str,
                "subset": sample.get("subset"),
                "question": raw_question,
                "gt_ans": gt_ans,
                "prediction": prediction,
                "correct": correct,
                "judge": judge_name,
                "hallucination_prediction": hallucination_pred,
                "detection_correct": detection_correct,
                "scores": {
                    "vl_uncertainty": result.uncertainty,
                    "cluster_ids": result.cluster_ids,
                    "cluster_distribution": result.cluster_distribution,
                },
                "samples": {
                    "sampled_answers": result.sampled_answers,
                    "perturbed_questions": result.perturbed_questions,
                    "most_likely_answer": result.most_likely_answer,
                },
                "config": {
                    "backend": args.backend,
                    "benchmark": args.benchmark,
                    "sampling_temp": args.sampling_temp,
                    "perturbation_type": args.perturbation_type,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            records.append(row)

    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    summary = _summarize(records)
    summary_path = Path(args.summary_output) if args.summary_output else out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False))


# ------------------------------------------------------------------
# Summary helpers
# ------------------------------------------------------------------
def _summarize(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    labeled = [r for r in records if r.get("correct") is not None]
    labels = [int(r["correct"]) for r in labeled]
    scores = [r["scores"]["vl_uncertainty"] for r in labeled]

    metrics_result = compute_metrics(labels, scores)

    detection = [r.get("detection_correct") for r in labeled if r.get("detection_correct") is not None]

    return {
        "n": len(records),
        "n_labeled": len(labels),
        "accuracy": metrics_result["accuracy"],
        "metrics": metrics_result["metrics"],
        "hallucination_detection_accuracy": float(np.mean(detection)) if detection else float("nan"),
    }


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id") is not None:
                done.add(str(row["id"]))
    return done


def _json_safe(value):
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _load_default_env_file():
    env_path = ROOT / "configs" / "vl_uncertainty.env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = os.path.expandvars(value.strip().strip("\"'"))
        if key and key not in os.environ:
            os.environ[key] = value


if __name__ == "__main__":
    main()
