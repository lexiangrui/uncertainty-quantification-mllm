#!/usr/bin/env python3
"""Run VAUQ self-evaluation on a local white-box LVLM.

Examples:
    python scripts/run_vauq.py --backend llava --benchmark cvbench --judge regex_choice
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
PROJECT_ROOT = ROOT.parents[1]
SRC = ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from judge import QwenLLMJudge, RegexChoiceJudge
from vauq.backends import build_backend
from vauq.benchmarks import build_benchmark
from vauq.eval import compute_metrics
from vauq.scoring import (
    compute_entropy_core_masked,
    compute_mask_comparison_scores,
    compute_multi_seed_comparison_scores,
    compute_vauq_scores,
)

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
    parser.add_argument(
        "--judge",
        choices=["regex_choice", "qwen_llm"],
        default=None,
        help="Project-wide judge; default is selected from the benchmark type.",
    )
    parser.add_argument("--model-path", default=os.environ.get("VAUQ_MODEL_PATH", "llava-hf/llava-1.5-7b-hf"))
    parser.add_argument("--attn-implementation", default="eager",
                        help="Use eager for attention-matrix VAUQ.")
    parser.add_argument("--topk-ratio", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument(
        "--mask-strategy", choices=["core", "random", "blank", "all"], default="core",
        help="all computes core/blank/random together from one answer and judge label.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--question-style", choices=["final_answer", "raw", "describe_then_answer"], default="final_answer",
        help="Use raw dataset questions for the hallucination-validity experiment.",
    )
    parser.add_argument(
        "--defer-judge", action="store_true",
        help="Generate/score now and leave correctness/hallucination labeling to a later multimodal judge.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "vauq" / "vauq_results.jsonl"))
    parser.add_argument("--summary-output", default=None)
    parser.add_argument(
        "--reference-results",
        default=None,
        help="Reuse prediction/correct/judge fields from a JSONL; generated answers must match.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-seeds",
        default=None,
        help="Comma-separated random-mask seeds computed together; requires reference results.",
    )
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
            row = json.loads(line)
            if row.get("id") is not None:
                done.add(str(row["id"]))
    return done


def load_reference_rows(path: str | None) -> dict[str, dict]:
    if path is None:
        return {}
    reference = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                reference[str(row["id"])] = row
    return reference


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)
    random_seeds = (
        [int(value) for value in args.random_seeds.split(",")]
        if args.random_seeds else []
    )
    if random_seeds and args.mask_strategy not in ("random", "all"):
        raise ValueError("--random-seeds requires --mask-strategy random or all")
    if random_seeds and args.mask_strategy == "random" and not args.reference_results:
        raise ValueError("random-only --random-seeds requires --reference-results")

    topk_ratio, alpha, layer_range = resolve_scoring_params(args)

    print(f"Loading backend={args.backend} model={args.model_path} attn={args.attn_implementation} ...")
    backend = build_backend(
        args.backend,
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
    )

    print(f"Loading benchmark={args.benchmark} ...")
    benchmark_kwargs = (
        {"question_style": args.question_style}
        if args.benchmark in {"mmvet", "vilp"} else {}
    )
    if args.question_style != "final_answer" and args.benchmark not in {"mmvet", "vilp"}:
        raise ValueError(f"--question-style {args.question_style} is only supported for MMVet and ViLP")
    benchmark = build_benchmark(args.benchmark, **benchmark_kwargs)

    reference_rows = load_reference_rows(args.reference_results)
    judge_name = args.judge or ("regex_choice" if args.benchmark == "cvbench" else "qwen_llm")
    expected_judge = "regex_choice" if args.benchmark == "cvbench" else "qwen_llm"
    if judge_name != expected_judge:
        raise ValueError(f"{args.benchmark} requires --judge {expected_judge}")
    judge = None if reference_rows or args.defer_judge else (
        RegexChoiceJudge() if args.benchmark == "cvbench"
        else QwenLLMJudge(os.environ["QWEN_JUDGE_MODEL"])
    )
    judge_description = (
        "reference_results" if reference_rows else
        "deferred_multimodal_judge" if args.defer_judge else type(judge).__name__
    )
    print(f"Using judge={judge_description} topk={topk_ratio} alpha={alpha} "
          f"layers={layer_range} mask={args.mask_strategy} decoding=greedy")

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
                raise ValueError(f"benchmark returned an incomplete sample at index {idx}")
            answer, generated_ids = backend.generate_with_ids(
                sample["img"], sample["question"],
                max_new_tokens=args.max_new_tokens,
            )
            reference = reference_rows.get(idx_str)
            if reference_rows:
                if reference is None:
                    raise ValueError(f"reference results missing sample {idx_str}")
                if answer != reference.get("prediction"):
                    raise ValueError(
                        f"generated answer differs from reference at {idx_str}: "
                        f"{answer!r} != {reference.get('prediction')!r}"
                    )
            if random_seeds and args.mask_strategy == "all":
                results = compute_multi_seed_comparison_scores(
                    backend, sample["img"], sample["question"], generated_ids,
                    random_seeds=random_seeds, topk_ratio=topk_ratio, alpha=alpha,
                    layer_range=layer_range, sample_index=idx, answer=answer,
                )
            elif random_seeds:
                reference_entropy = reference["scores"]["core"]["entropy"]
                results = {}
                for random_seed in random_seeds:
                    entropy_masked = compute_entropy_core_masked(
                        backend, sample["img"], sample["question"], generated_ids,
                        topk_ratio, layer_range, mask_strategy="random",
                        mask_seed=random_seed + idx,
                    )
                    is_score = entropy_masked - reference_entropy
                    results[f"random_seed{random_seed}"] = {
                        "entropy": reference_entropy,
                        "entropy_masked": entropy_masked,
                        "is_score": is_score,
                        "vauq": reference_entropy - alpha * is_score,
                    }
            elif args.mask_strategy == "all":
                results = compute_mask_comparison_scores(
                    backend, sample["img"], sample["question"], generated_ids,
                    topk_ratio=topk_ratio, alpha=alpha, layer_range=layer_range,
                    mask_seed=args.seed + idx, answer=answer,
                )
            else:
                result = compute_vauq_scores(
                    backend, sample["img"], sample["question"], generated_ids,
                    topk_ratio=topk_ratio, alpha=alpha, layer_range=layer_range,
                    mask_strategy=args.mask_strategy,
                    mask_seed=args.seed + idx,
                    answer=answer,
                )
            if args.defer_judge:
                correct = None
                judge_result = None
            elif reference is not None:
                correct = reference.get("correct")
                judge_result = reference.get("judge_result")
            elif args.benchmark == "cvbench":
                gold_letter = str(sample["gt_ans"]).strip().strip("()").upper()
                gold_index = ord(gold_letter) - ord("A")
                correct = judge.judge(answer, gold_index, sample["choices"], mode="letter")
                judge_result = getattr(judge, "last_result", None)
            else:
                correct = judge.judge(sample["question"], [str(sample.get("gt_ans"))], answer)
                judge_result = getattr(judge, "last_result", None)
            row = {
                "id": idx_str,
                "subset": sample.get("subset"),
                "question": sample["question"],
                "gt_ans": sample.get("gt_ans"),
                "prediction": answer,
                "generated_ids": generated_ids[0].detach().cpu().tolist(),
                "correct": correct,
                "judge": judge_name,
                "judge_result": judge_result,
                "scores": (
                    {
                        strategy: {
                            "entropy": value.entropy,
                            "entropy_masked": value.entropy_masked,
                            "is_score": value.is_score,
                            "vauq": value.vauq,
                        }
                        for strategy, value in results.items()
                    }
                    if random_seeds and args.mask_strategy == "all"
                    else results
                    if random_seeds
                    else
                    {
                        strategy: {
                            "entropy": value.entropy,
                            "entropy_masked": value.entropy_masked,
                            "is_score": value.is_score,
                            "vauq": value.vauq,
                        }
                        for strategy, value in results.items()
                    }
                    if args.mask_strategy == "all"
                    else {
                        "entropy": result.entropy,
                        "entropy_masked": result.entropy_masked,
                        "is_score": result.is_score,
                        "vauq": result.vauq,
                    }
                ),
                "config": {
                    "backend": args.backend, "benchmark": args.benchmark, "judge": judge_name,
                    "model_path": args.model_path,
                    "attn_implementation": args.attn_implementation,
                    "topk_ratio": topk_ratio,
                    "alpha": alpha, "layer_range": list(layer_range),
                    "mask_strategy": args.mask_strategy,
                    "mask_seed": (
                        args.seed + idx
                        if args.mask_strategy in ("random", "all") else None
                    ),
                    "random_seeds": random_seeds or None,
                    "decoding": "greedy",
                    "max_new_tokens": args.max_new_tokens,
                    "question_style": args.question_style,
                    "defer_judge": args.defer_judge,
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
        summary["per_subset"] = {
            s: _summarize_group(rs)
            for s, rs in sorted(groups.items())
        }
    return summary


def _summarize_group(records: list[dict]) -> dict:
    labeled = [r for r in records if r.get("correct") is not None]
    labels = [int(r["correct"]) for r in labeled]
    if labeled and "core" in labeled[0]["scores"]:
        strategies = list(labeled[0]["scores"])
        methods = {
            strategy: _method_metrics(labels, labeled, strategy)
            for strategy in strategies
        }
        random_names = [name for name in strategies if name.startswith("random_seed")]
        return {
            "n": len(records),
            "n_labeled": len(labels),
            "accuracy": sum(labels) / len(labels) if labels else float("nan"),
            "methods": methods,
            "random_aggregate": _aggregate_random_metrics(methods, random_names),
        }
    if labeled and all(name.startswith("random_seed") for name in labeled[0]["scores"]):
        return {
            "n": len(records),
            "n_labeled": len(labels),
            "accuracy": sum(labels) / len(labels) if labels else float("nan"),
            "methods": {
                strategy: _method_metrics(labels, labeled, strategy)
                for strategy in labeled[0]["scores"]
            },
        }
    return _method_metrics(labels, labeled)


def _method_metrics(
    labels: list[int], records: list[dict], strategy: str | None = None
) -> dict:
    def score(record, name):
        values = record["scores"] if strategy is None else record["scores"][strategy]
        return values[name]

    vauq = [score(r, "vauq") for r in records]
    entropy = [score(r, "entropy") for r in records]
    is_score = [score(r, "is_score") for r in records]
    if len(set(labels)) < 2:
        # A tiny smoke subset can contain only correct or only incorrect
        # answers. The inference output is still valid; AUROC/AUPR are not.
        undefined = {"auroc": float("nan"), "aupr": float("nan")}
        metric_values = {
            "vauq": dict(undefined),
            "entropy": dict(undefined),
            "is_score": dict(undefined),
        }
        accuracy = sum(labels) / len(labels) if labels else float("nan")
    else:
        metrics = compute_metrics(labels, vauq, entropy, is_score)
        metric_values = metrics["metrics"]
        accuracy = metrics["accuracy"]
    return {
        "accuracy": accuracy,
        "metrics": metric_values,
    }


def _aggregate_random_metrics(methods: dict, names: list[str]) -> dict | None:
    if not names:
        return None
    aggregate = {"seeds": [int(name.removeprefix("random_seed")) for name in names]}
    for score_name in ("vauq", "entropy", "is_score"):
        aggregate[score_name] = {}
        for metric_name in ("auroc", "aupr"):
            values = [methods[name]["metrics"][score_name][metric_name] for name in names]
            aggregate[score_name][metric_name] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "values": values,
            }
    return aggregate


if __name__ == "__main__":
    main()
