#!/usr/bin/env python3
"""Run VAUQ self-evaluation on a local white-box LVLM.

Examples:
    python scripts/run_vauq.py --backend llava --benchmark cvbench --judge letter
    python scripts/run_vauq.py --backend llava --benchmark cvbench --limit 4
"""

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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vauq.backends import build_backend
from vauq.benchmarks import build_benchmark
from vauq.eval import compute_metrics
from vauq.judges import DEFAULT_JUDGE, build_judge
from vauq.scoring import compute_vauq_scores

# Per-(model, benchmark) defaults from the VAUQ paper Appendix F.
# Values are keyed by the lower-cased basename of the local model path.
DEFAULT_HYPERPARAMETERS: dict[tuple[str, str], dict[str, object]] = {
    ("llava-1.5-7b-hf", "vilp"): {
        "topk_ratio": 0.6, "alpha": 0.6, "layer_range": (10, 25)
    },
    ("llava-1.5-7b-hf", "mmvet"): {
        "topk_ratio": 0.4, "alpha": 0.6, "layer_range": (10, 25)
    },
    ("llava-1.5-7b-hf", "cvbench"): {
        "topk_ratio": 0.3, "alpha": 1.2, "layer_range": (10, 25)
    },
    ("llava-1.5-13b-hf", "vilp"): {
        "topk_ratio": 0.2, "alpha": 1.5, "layer_range": (10, 35)
    },
    ("llava-1.5-13b-hf", "mmvet"): {
        "topk_ratio": 0.3, "alpha": 0.4, "layer_range": (10, 35)
    },
    ("llava-1.5-13b-hf", "cvbench"): {
        "topk_ratio": 0.4, "alpha": 1.2, "layer_range": (10, 35)
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VAUQ on a local LVLM.")
    parser.add_argument("--backend", choices=["llava"], default="llava")
    parser.add_argument("--benchmark", choices=["mmvet", "cvbench", "vilp"], default="cvbench")
    parser.add_argument("--judge", choices=["letter", "llm", "qwen_local", "none"], default=None,
                        help="Default is per-benchmark (cvbench->letter, free-form->qwen_local).")
    parser.add_argument("--model-path", default=os.environ.get("VAUQ_MODEL_PATH", "llava-hf/llava-1.5-7b-hf"))
    parser.add_argument("--topk-ratio", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--inference-temp", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mask-strategy", choices=["core", "blank"], default="core")
    parser.add_argument("--ablation-baseline", choices=["attention_mask", "mean"], default="attention_mask")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", default="results/vauq_results.jsonl")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_short_name(model_path: str) -> str:
    return os.path.basename(model_path.rstrip("/")).lower()


def resolve_scoring_params(args: argparse.Namespace) -> tuple[float, float, tuple[int, int]]:
    key = (model_short_name(args.model_path), args.benchmark)
    defaults = DEFAULT_HYPERPARAMETERS.get(key)
    required = (
        args.topk_ratio,
        args.alpha,
        args.layer_start,
        args.layer_end,
    )
    if defaults is None and any(v is None for v in required):
        raise SystemExit(
            f"No default hyperparameters for model={key[0]!r}, benchmark={args.benchmark!r}. "
            "Pass --topk-ratio, --alpha, --layer-start, and --layer-end explicitly."
        )
    defaults = defaults or {}
    default_layer = defaults.get("layer_range", (10, 25))
    topk_ratio = args.topk_ratio if args.topk_ratio is not None else float(defaults["topk_ratio"])
    alpha = args.alpha if args.alpha is not None else float(defaults["alpha"])
    layer_start = args.layer_start if args.layer_start is not None else int(default_layer[0])
    layer_end = args.layer_end if args.layer_end is not None else int(default_layer[1])
    print(
        f"Using scoring params for {key[0]}/{args.benchmark}: "
        f"topk_ratio={topk_ratio}, alpha={alpha}, layers=({layer_start}, {layer_end})"
    )
    return topk_ratio, alpha, (layer_start, layer_end)


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

    topk_ratio, alpha, layer_range = resolve_scoring_params(args)

    print(f"Loading backend={args.backend} model={args.model_path} ...")
    backend = build_backend(args.backend, model_path=args.model_path)

    print(f"Loading benchmark={args.benchmark} ...")
    benchmark = build_benchmark(args.benchmark)

    judge_name = args.judge if args.judge is not None else DEFAULT_JUDGE.get(args.benchmark, "none")
    judge = build_judge(judge_name, benchmark=args.benchmark)
    print(f"Using judge={type(judge).__name__} mask={args.mask_strategy} "
          f"baseline={args.ablation_baseline} topk={topk_ratio} alpha={alpha} "
          f"layers={layer_range} temp={args.inference_temp}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(out_path) if args.resume else set()
    mode = "a" if args.resume and done else "w"

    size = benchmark.obtain_size()
    limit = None if args.limit is None or args.limit <= 0 else args.limit
    end = size if limit is None else min(size, args.start + limit)
    records: list[dict] = []

    with out_path.open(mode, encoding="utf-8") as f:
        for idx in tqdm(range(args.start, end), desc=f"VAUQ {args.benchmark}"):
            idx_str = str(idx)
            if idx_str in done:
                continue
            sample = benchmark.retrieve(idx)
            if sample is None or sample.get("img") is None:
                continue
            answer, generated_ids = backend.generate_with_ids(
                sample["img"], sample["question"],
                temp=args.inference_temp, max_new_tokens=args.max_new_tokens,
            )
            result = compute_vauq_scores(
                backend, sample["img"], sample["question"], generated_ids,
                topk_ratio=topk_ratio, alpha=alpha, layer_range=layer_range,
                mask_strategy=args.mask_strategy, ablation_baseline=args.ablation_baseline,
                answer=answer,
            )
            correct = judge.judge(answer, sample.get("gt_ans"), sample)
            judge_result = getattr(judge, "last_result", None)
            row = {
                "id": idx_str,
                "subset": sample.get("subset"),
                "question": sample["question"],
                "gt_ans": sample.get("gt_ans"),
                "prediction": answer,
                "correct": correct,
                "judge": judge_name,
                "judge_result": judge_result,
                "scores": {
                    "entropy": result.entropy,
                    "entropy_masked": result.entropy_masked,
                    "is_score": result.is_score,
                    "vauq": result.vauq,
                },
                "config": {
                    "backend": args.backend, "benchmark": args.benchmark, "judge": judge_name,
                    "model_path": args.model_path, "topk_ratio": topk_ratio,
                    "alpha": alpha, "layer_range": list(layer_range),
                    "mask_strategy": args.mask_strategy,
                    "ablation_baseline": args.ablation_baseline,
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

    summary = _summarize(records)
    summary_path = Path(args.summary_output) if args.summary_output else out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _summarize(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    summary = _summarize_group(records)
    summary["n"] = len(records)

    # per-subset breakdown (e.g. CV-Bench 2D / 3D)
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("subset") or "all", []).append(r)
    if len(groups) > 1:
        summary["per_subset"] = {s: _summarize_group(rs) for s, rs in sorted(groups.items())}
    return summary


def _summarize_group(records: list[dict]) -> dict:
    labeled = [r for r in records if r.get("correct") is not None]
    labels = [int(r["correct"]) for r in labeled]
    vauq = [r["scores"]["vauq"] for r in labeled]
    entropy = [r["scores"]["entropy"] for r in labeled]
    is_score = [r["scores"]["is_score"] for r in labeled]
    metrics = compute_metrics(labels, vauq, entropy, is_score)
    return {
        "n": len(records),
        "n_labeled": len(labels),
        "accuracy": metrics["accuracy"],
        "metrics": metrics["metrics"],
    }


if __name__ == "__main__":
    main()
