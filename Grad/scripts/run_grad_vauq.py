#!/usr/bin/env python3
"""Run Grad-VAUQ self-evaluation on a local white-box LVLM."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "baseline" / "vauq-repro" / "src"
GRAD_ROOT = ROOT / "Grad"
for path in (SRC, GRAD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grad_vauq.backends import build_backend
from grad_vauq.scoring import compute_grad_vauq_scores
from vauq.benchmarks import build_benchmark
from vauq.eval import compute_metrics
from vauq.judges import DEFAULT_JUDGE, build_judge


DEFAULT_HYPERPARAMETERS: dict[tuple[str, str], dict[str, float]] = {
    ("llava-1.5-7b-hf", "vilp"): {"topk_ratio": 0.6, "alpha": 0.6},
    ("llava-1.5-7b-hf", "mmvet"): {"topk_ratio": 0.4, "alpha": 0.6},
    ("llava-1.5-7b-hf", "cvbench"): {"topk_ratio": 0.3, "alpha": 1.2},
    ("llava-1.5-13b-hf", "vilp"): {"topk_ratio": 0.2, "alpha": 1.5},
    ("llava-1.5-13b-hf", "mmvet"): {"topk_ratio": 0.3, "alpha": 0.4},
    ("llava-1.5-13b-hf", "cvbench"): {"topk_ratio": 0.4, "alpha": 1.2},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grad-VAUQ on a local LVLM.")
    parser.add_argument("--backend", choices=["llava"], default="llava")
    parser.add_argument("--visual-adapter", default="llava")
    parser.add_argument("--benchmark", choices=["mmvet", "cvbench", "vilp"], default="cvbench")
    parser.add_argument("--judge", choices=["letter", "llm", "qwen_local", "none"], default=None)
    parser.add_argument("--model-path", default=os.environ.get("VAUQ_MODEL_PATH", "llava-hf/llava-1.5-7b-hf"))
    parser.add_argument("--attn-implementation", default="flash_attention_2",
                        help="Use flash_attention_2 by default; pass sdpa if flash-attn is unavailable.")
    parser.add_argument("--selector", choices=["grad_x_act", "integrated_gradients"], default="grad_x_act")
    parser.add_argument("--attribution-baseline", choices=["zero", "mean"], default="mean",
                        help="Baseline used by attribution methods such as integrated gradients.")
    parser.add_argument("--ig-steps", type=int, default=16,
                        help="Number of interpolation steps for integrated gradients.")
    parser.add_argument("--ablation-baseline", choices=["zero", "mean"], default="zero")
    parser.add_argument("--topk-ratio", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--inference-temp", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", default="results/grad_vauq_results.jsonl")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--store-visual-scores", action="store_true",
                        help="Store full per-visual-token scores in JSONL. Large outputs.")
    return parser.parse_args()


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_short_name(model_path: str) -> str:
    return os.path.basename(model_path.rstrip("/")).lower()


def resolve_scoring_params(args: argparse.Namespace) -> tuple[float, float]:
    key = (model_short_name(args.model_path), args.benchmark)
    defaults = DEFAULT_HYPERPARAMETERS.get(key)
    if defaults is None and (args.topk_ratio is None or args.alpha is None):
        raise SystemExit(
            f"No defaults for model={key[0]!r}, benchmark={args.benchmark!r}. "
            "Pass --topk-ratio and --alpha explicitly."
        )
    defaults = defaults or {}
    topk_ratio = args.topk_ratio if args.topk_ratio is not None else float(defaults["topk_ratio"])
    alpha = args.alpha if args.alpha is not None else float(defaults["alpha"])
    print(f"Using Grad-VAUQ params for {key[0]}/{args.benchmark}: topk_ratio={topk_ratio}, alpha={alpha}")
    return topk_ratio, alpha


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
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


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)
    topk_ratio, alpha = resolve_scoring_params(args)

    print(
        f"Loading backend={args.backend} model={args.model_path} "
        f"attn={args.attn_implementation} adapter={args.visual_adapter} ..."
    )
    backend = build_backend(
        args.backend,
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        adapter=args.visual_adapter,
    )

    print(f"Loading benchmark={args.benchmark} ...")
    benchmark = build_benchmark(args.benchmark)
    judge_name = args.judge if args.judge is not None else DEFAULT_JUDGE.get(args.benchmark, "none")
    judge = build_judge(judge_name, benchmark=args.benchmark)
    print(
        f"Using judge={type(judge).__name__} selector={args.selector} "
        f"baseline={args.ablation_baseline} topk={topk_ratio} alpha={alpha}"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path) if args.resume else set()
    mode = "a" if args.resume and done else "w"

    size = benchmark.obtain_size()
    limit = None if args.limit is None or args.limit <= 0 else args.limit
    end = size if limit is None else min(size, args.start + limit)
    records: list[dict] = []

    with out_path.open(mode, encoding="utf-8") as f:
        for idx in tqdm(range(args.start, end), desc=f"Grad-VAUQ {args.benchmark}"):
            idx_str = str(idx)
            if idx_str in done:
                continue
            sample = benchmark.retrieve(idx)
            if sample is None or sample.get("img") is None:
                continue
            answer, generated_ids = backend.generate_with_ids(
                sample["img"],
                sample["question"],
                temp=args.inference_temp,
                max_new_tokens=args.max_new_tokens,
            )
            result = compute_grad_vauq_scores(
                backend,
                sample["img"],
                sample["question"],
                generated_ids,
                topk_ratio=topk_ratio,
                alpha=alpha,
                selector_name=args.selector,
                ablation_baseline=args.ablation_baseline,
                attribution_baseline=args.attribution_baseline,
                ig_steps=args.ig_steps,
                answer=answer,
                store_visual_scores=args.store_visual_scores,
            )
            correct = judge.judge(answer, sample.get("gt_ans"), sample)
            judge_result = getattr(judge, "last_result", None)
            row = {
                "id": idx_str,
                "subset": sample.get("subset"),
                "question": sample["question"],
                "gt_ans": sample.get("gt_ans"),
                "prediction": answer,
                "generated_ids": generated_ids.detach().cpu().tolist()[0],
                "correct": correct,
                "judge": judge_name,
                "judge_result": judge_result,
                "scores": {
                    "entropy": result.entropy,
                    "entropy_masked": result.entropy_masked,
                    "is_score": result.is_score,
                    "vauq": result.vauq,
                },
                "grad": {
                    "selected_indices": result.selected_indices,
                    "spatial_shape": list(result.spatial_shape) if result.spatial_shape else None,
                    "visual_scores": result.visual_scores,
                },
                "config": {
                    "method": "grad_vauq",
                    "backend": args.backend,
                    "visual_adapter": args.visual_adapter,
                    "benchmark": args.benchmark,
                    "judge": judge_name,
                    "model_path": args.model_path,
                    "attn_implementation": args.attn_implementation,
                    "selector": args.selector,
                    "attribution_baseline": args.attribution_baseline,
                    "ig_steps": args.ig_steps,
                    "ablation_baseline": args.ablation_baseline,
                    "topk_ratio": topk_ratio,
                    "alpha": alpha,
                    "temp": args.inference_temp,
                    "max_new_tokens": args.max_new_tokens,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            records.append(row)

    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    summary = summarize(records)
    summary_path = Path(args.summary_output) if args.summary_output else out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    summary = summarize_group(records)
    summary["n"] = len(records)
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record.get("subset") or "all", []).append(record)
    if len(groups) > 1:
        summary["per_subset"] = {
            subset: summarize_group(rows) for subset, rows in sorted(groups.items())
        }
    return summary


def summarize_group(records: list[dict]) -> dict:
    labeled = [record for record in records if record.get("correct") is not None]
    labels = [int(record["correct"]) for record in labeled]
    vauq = [record["scores"]["vauq"] for record in labeled]
    entropy = [record["scores"]["entropy"] for record in labeled]
    is_score = [record["scores"]["is_score"] for record in labeled]
    metrics = compute_metrics(labels, vauq, entropy, is_score)
    return {
        "n": len(records),
        "n_labeled": len(labels),
        "accuracy": metrics["accuracy"],
        "metrics": metrics["metrics"],
    }


if __name__ == "__main__":
    main()
