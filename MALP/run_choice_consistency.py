"""Generate multiple latent-perturbed answers for multiple-choice samples."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LLAVA_MODEL,
    NUM_CONSISTENCY_GENERATIONS,
    PERTURB_ROOT,
    PERTURB_SIGMA,
    REASONING_LAYERS,
    SEED,
    TEXT_GAMMA,
)
from data import iter_samples
from io_utils import append_jsonl, load_jsonl_by_id
from model import LlavaMalpRunner
from perturb import PerturbSpec


LOGGER = logging.getLogger("malp.run_choice_consistency")
SUPPORTED_MODES = ("norm_isotropic", "directional")


def process_one(
    sample: dict,
    runner: LlavaMalpRunner,
    *,
    modes: list[str],
    stages: list[str],
    seeds: tuple[int, ...],
    sigma: float,
    gamma: float,
    experiment_config: dict,
) -> dict:
    choices = sample.get("choices")
    answer_index = sample.get("answer_index")
    if not choices or answer_index is None:
        raise ValueError(f"sample {sample.get('id')!r} is not multiple choice")
    started = time.perf_counter()
    inputs = runner.prepare_inputs(sample["image"], sample["question"])
    base = runner.greedy_generate(inputs)
    generations = []
    for stage in stages:
        for mode in modes:
            for seed in seeds:
                spec = PerturbSpec(
                    modality="joint",
                    stage=stage,
                    mode=mode,
                    sigma=sigma,
                    gamma=gamma,
                    seed=seed,
                )
                result = runner.generate_with_perturbation(inputs, spec)
                generations.append(
                    {
                        "stage": stage,
                        "mode": mode,
                        "modality": "joint",
                        "layers": list(REASONING_LAYERS) if stage == "reasoning" else None,
                        "sigma": sigma,
                        "gamma": gamma,
                        "seed": seed,
                        "text": result["text"],
                        "answer_ids": result["answer_ids"].detach().cpu().reshape(-1).tolist(),
                    }
                )
    return {
        "id": sample["id"],
        "dataset": sample["dataset"],
        "question": sample["question"],
        "prediction": base["text"],
        "references": sample["references"],
        "choices": choices,
        "answer_index": answer_index,
        "generations": generations,
        "metadata": sample["metadata"],
        "runtime_seconds": time.perf_counter() - started,
        "experiment_config": experiment_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cvbench", "cvbench2d"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="process every Nth dataset item (before --limit), for deterministic broad scans",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="zero-based offset within --sample-stride",
    )
    parser.add_argument(
        "--output", type=Path, default=PERTURB_ROOT / "choice_consistency.jsonl"
    )
    parser.add_argument(
        "--modes", nargs="+", default=list(SUPPORTED_MODES), choices=SUPPORTED_MODES
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["fusion", "reasoning"],
        choices=["fusion", "reasoning"],
    )
    parser.add_argument(
        "--num-generations", type=int, default=NUM_CONSISTENCY_GENERATIONS
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sigma", type=float, default=PERTURB_SIGMA)
    parser.add_argument("--gamma", type=float, default=TEXT_GAMMA)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.num_generations < 2:
        raise ValueError("num-generations must be at least 2")
    if args.sample_stride < 1:
        raise ValueError("sample-stride must be at least 1")
    if not 0 <= args.sample_offset < args.sample_stride:
        raise ValueError("sample-offset must be in [0, sample-stride)")
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {args.output}")
    seeds = tuple(args.seed + index for index in range(args.num_generations))
    experiment_config = {
        "schema_version": 3,
        "uncertainty_method": "choice_answer_consistency",
        "decode_strategy": "greedy",
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "sample_stride": args.sample_stride,
        "sample_offset": args.sample_offset,
        "model": str(LLAVA_MODEL),
        "stages": args.stages,
        "reasoning_layers": list(REASONING_LAYERS),
        "modes": args.modes,
        "seeds": list(seeds),
        "sigma": args.sigma,
        "gamma": args.gamma,
    }
    existing = load_jsonl_by_id(args.output) if args.resume else {}
    for record_id, record in existing.items():
        if record.get("experiment_config") != experiment_config:
            raise ValueError(f"resume configuration mismatch for {record_id!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner = LlavaMalpRunner()
    processed = len(existing)
    with args.output.open("a", encoding="utf-8") as handle:
        emitted = 0
        for dataset_index, sample in enumerate(
            iter_samples(args.dataset, split=args.split)
        ):
            if dataset_index % args.sample_stride != args.sample_offset:
                continue
            if args.limit is not None and emitted >= args.limit:
                break
            emitted += 1
            if sample["id"] in existing:
                continue
            record = process_one(
                sample,
                runner,
                modes=args.modes,
                stages=args.stages,
                seeds=seeds,
                sigma=args.sigma,
                gamma=args.gamma,
                experiment_config=experiment_config,
            )
            append_jsonl(handle, record)
            processed += 1
            LOGGER.info(
                "processed=%d id=%s prediction=%r generations=%d seconds=%.3f",
                processed,
                record["id"],
                record["prediction"],
                len(record["generations"]),
                record["runtime_seconds"],
            )
    LOGGER.info("complete processed=%d output=%s", processed, args.output)


if __name__ == "__main__":
    main()
