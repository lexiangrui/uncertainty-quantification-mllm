import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LLAVA_MODEL,
    NUM_PERTURBATIONS,
    PERTURB_ROOT,
    PERTURB_SIGMA,
    PERTURBATION_SEEDS,
    REASONING_LAYERS,
    TEXT_GAMMA,
)
from data import iter_samples
from io_utils import append_jsonl, load_jsonl_by_id
from model import LlavaMalpRunner
from perturb import PerturbSpec


LOGGER = logging.getLogger("malp.run_perturb")


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
    started = time.perf_counter()
    inputs = runner.prepare_inputs(sample["image"], sample["question"])
    base = runner.greedy_generate(inputs)
    teacher_inputs = runner.build_teacher_forcing_inputs(
        inputs, base["answer_ids"], base["answer_mask"]
    )
    answer_ids = teacher_inputs["answer_ids"]
    if not torch.is_tensor(answer_ids):
        raise TypeError("answer_ids must be a tensor")
    answer_mask = teacher_inputs["answer_mask"]
    adversarial_gradients: dict[str, torch.Tensor] = {}
    original_logits = None
    if "adversarial" in modes:
        for stage in stages:
            gradient_result = runner.compute_logprob_gradient(teacher_inputs, stage)
            adversarial_gradients[stage] = gradient_result["gradient"]
            if original_logits is None:
                original_logits = gradient_result["response_logits"]
    if original_logits is None:
        original_logits = runner.forward_original(teacher_inputs)["response_logits"]
    nll0 = runner.mean_nll(original_logits, answer_ids, answer_mask)

    perturbation_records = []
    for stage in stages:
        for mode in modes:
            adv_gradient = None
            if mode == "adversarial":
                adv_gradient = adversarial_gradients[stage]
            for seed in seeds:
                spec = PerturbSpec(
                    modality="joint",
                    stage=stage,
                    mode=mode,
                    sigma=sigma,
                    gamma=gamma,
                    seed=seed,
                    adv_gradient=adv_gradient,
                )
                result = runner.forward_with_perturbation(teacher_inputs, spec)
                perturbed_logits = result["response_logits"]
                perturbation_records.append(
                    {
                        "stage": stage,
                        "modality": "joint",
                        "layers": list(REASONING_LAYERS) if stage == "reasoning" else None,
                        "mode": mode,
                        "sigma": sigma,
                        "gamma": gamma,
                        "seed": seed,
                        "nll": runner.mean_nll(perturbed_logits, answer_ids, answer_mask),
                        "kl": runner.mean_kl(original_logits, perturbed_logits, answer_mask),
                    }
                )

    return {
        "id": sample["id"],
        "dataset": sample["dataset"],
        "question": sample["question"],
        "prediction": base["text"],
        "references": sample["references"],
        "choices": sample["choices"],
        "answer_index": sample["answer_index"],
        "answer_ids": answer_ids.detach().cpu().reshape(-1).tolist(),
        "prompt_length": int(teacher_inputs["prompt_length"]),
        "answer_length": int(teacher_inputs["answer_length"]),
        "nll0": nll0,
        "perturbations": perturbation_records,
        "metadata": sample["metadata"],
        "runtime_seconds": time.perf_counter() - started,
        "experiment_config": experiment_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cvbench", "cvbench2d", "mmvet", "vilp"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=PERTURB_ROOT / "perturb.jsonl")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["norm_isotropic", "directional", "adversarial"],
        choices=["norm_isotropic", "directional", "adversarial"],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["fusion", "reasoning"],
        choices=["fusion", "reasoning"],
    )
    parser.add_argument("--num-perturbations", type=int, default=NUM_PERTURBATIONS)
    parser.add_argument("--sigma", type=float, default=PERTURB_SIGMA)
    parser.add_argument("--gamma", type=float, default=TEXT_GAMMA)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not 1 <= args.num_perturbations <= len(PERTURBATION_SEEDS):
        raise ValueError(f"num_perturbations must be in [1, {len(PERTURBATION_SEEDS)}]")
    seeds = tuple(PERTURBATION_SEEDS[: args.num_perturbations])
    gamma = args.gamma
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {args.output}")
    experiment_config = {
        "schema_version": 2,
        "dataset": args.dataset,
        "split": args.split,
        "model": str(LLAVA_MODEL),
        "stages": args.stages,
        "reasoning_layers": list(REASONING_LAYERS),
        "modes": args.modes,
        "seeds": list(seeds),
        "sigma": args.sigma,
        "gamma": gamma,
    }
    existing = load_jsonl_by_id(args.output) if args.resume else {}
    for record_id, record in existing.items():
        saved_config = record.get("experiment_config")
        if saved_config is None:
            raise ValueError(
                f"cannot safely resume legacy record {record_id!r} without experiment_config"
            )
        if saved_config != experiment_config:
            raise ValueError(
                f"resume configuration mismatch for {record_id!r}: "
                f"saved={saved_config!r}, requested={experiment_config!r}"
            )
    done = set(existing)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "start dataset=%s stages=%s modes=%s seeds=%s sigma=%.6f gamma=%.6f output=%s completed=%d",
        args.dataset,
        args.stages,
        args.modes,
        seeds,
        args.sigma,
        gamma,
        args.output,
        len(done),
    )
    runner = LlavaMalpRunner()
    processed = len(done)
    with args.output.open("a", encoding="utf-8") as handle:
        for sample in iter_samples(args.dataset, split=args.split, limit=args.limit):
            if sample["id"] in done:
                continue
            record = process_one(
                sample,
                runner,
                modes=args.modes,
                stages=args.stages,
                seeds=seeds,
                sigma=args.sigma,
                gamma=gamma,
                experiment_config=experiment_config,
            )
            append_jsonl(handle, record)
            processed += 1
            LOGGER.info(
                "processed=%d id=%s prediction=%r nll0=%.4f perturbations=%d seconds=%.3f",
                processed,
                record["id"],
                record["prediction"],
                record["nll0"],
                len(record["perturbations"]),
                record["runtime_seconds"],
            )
    LOGGER.info("complete processed=%d output=%s", processed, args.output)


if __name__ == "__main__":
    main()
