import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch

from config import (
    METHOD_VERSION,
    NORM_ISOTROPIC_SIGMA,
    PERTURBATION_SEEDS,
    SEED,
    SEMANTIC_VOLUME_JITTER,
    SENSITIVE_TOKEN_RATIO,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from judge import RegexChoiceJudge

from data import iter_samples
from io_utils import append_jsonl, load_jsonl_by_id
from model import LlavaRunner
from perturb import (
    PERTURBATION_MODES,
    combined_uncertainty,
    nll_instability,
    select_sensitive_indices,
    semantic_volume,
    visual_dependency_scores,
)


LOGGER = logging.getLogger("gasp.run")


def completed_ids(
    path: Path,
    *,
    perturbation_mode: str,
    norm_isotropic_sigma: float,
) -> set[str]:
    records = load_jsonl_by_id(path)
    expected_perturbation = {
        "mode": perturbation_mode,
        "norm_isotropic_sigma": (
            norm_isotropic_sigma if perturbation_mode == "norm_isotropic" else None
        ),
    }
    incompatible = []
    for record_id, record in records.items():
        if record.get("method_version") != METHOD_VERSION:
            incompatible.append(record_id)
            continue
        if record.get("perturbation") != expected_perturbation:
            incompatible.append(record_id)
    if incompatible:
        raise ValueError(
            f"resume file contains {len(incompatible)} records from an incompatible GASP version; "
            "use a new output path"
        )
    return set(records)


def _perturbed_runs(
    llava: LlavaRunner,
    inputs: dict[str, torch.Tensor],
    answer_ids: torch.Tensor,
    selected_indices: torch.Tensor,
    reference_indices: torch.Tensor,
    perturbation_mode: str,
    norm_isotropic_sigma: float,
) -> list[dict]:
    return [
        llava.score_fixed_answer(
            inputs,
            answer_ids,
            replacement_indices=selected_indices,
            reference_indices=reference_indices,
            replacement_seed=seed,
            perturbation_mode=perturbation_mode,
            norm_isotropic_sigma=norm_isotropic_sigma,
            return_last_hidden=True,
        )
        for seed in PERTURBATION_SEEDS
    ]


def process_one(
    sample: dict,
    llava: LlavaRunner,
    *,
    perturbation_mode: str,
    norm_isotropic_sigma: float,
) -> dict:
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    inputs = llava.prepare_inputs(sample["image"], sample["question"])
    base = llava.greedy_generate(inputs)

    traced = llava.score_fixed_answer(inputs, base["answer_ids"], capture_embedding_grad=True)
    nll0 = traced["mean_nll"]
    visual_indices = select_sensitive_indices(
        traced["gradient_scores"], traced["visual_mask"], SENSITIVE_TOKEN_RATIO
    )
    visual_reference = torch.nonzero(traced["visual_mask"], as_tuple=True)[0]

    visual_runs = _perturbed_runs(
        llava,
        inputs,
        base["answer_ids"],
        visual_indices,
        visual_reference,
        perturbation_mode,
        norm_isotropic_sigma,
    )
    visual_nlls = [run["mean_nll"] for run in visual_runs]
    visual_nll_result = nll_instability(nll0, visual_nlls)
    visual_volume = semantic_volume(
        [run["last_hidden"] for run in visual_runs], SEMANTIC_VOLUME_JITTER
    )

    visual_nll_dependency = visual_nll_result["score"]
    visual_volume_dependency = visual_volume["score"]
    dependency_result = visual_dependency_scores(
        visual_nll_dependency, visual_volume_dependency
    )
    visual_dependency = dependency_result["visual_dependency"]
    visual_ungrounded_risk = dependency_result["visual_ungrounded_risk"]
    predictive_uncertainty = 1.0 - math.exp(-nll0)
    uncertainty = combined_uncertainty(predictive_uncertainty, visual_ungrounded_risk)
    values = [
        visual_nll_dependency,
        visual_volume_dependency,
        visual_dependency,
        visual_ungrounded_risk,
        predictive_uncertainty,
        uncertainty,
    ]
    if not all(0.0 <= value <= 1.0 for value in values):
        raise AssertionError(f"score outside [0,1]: {values}")
    correct = (
        RegexChoiceJudge().judge(
            base["text"],
            sample["answer_index"],
            sample["choices"],
            mode=sample.get("judge_mode", "letter"),
        )
        if sample["choices"]
        else None
    )
    return {
        "method_version": METHOD_VERSION,
        "perturbation": {
            "mode": perturbation_mode,
            "norm_isotropic_sigma": (
                norm_isotropic_sigma if perturbation_mode == "norm_isotropic" else None
            ),
        },
        "id": sample["id"],
        "question": sample["question"],
        "prediction": base["text"],
        "references": sample["references"],
        "correct": correct,
        "judge": RegexChoiceJudge.name if sample["choices"] else None,
        "error_label": None if correct is None else int(not correct),
        "scores": {
            "nll0": nll0,
            "visual_nll_dependency": visual_nll_dependency,
            "visual_volume_dependency": visual_volume_dependency,
            "visual_dependency": visual_dependency,
            "predictive_uncertainty": predictive_uncertainty,
            "visual_ungrounded_risk": visual_ungrounded_risk,
            "uncertainty": uncertainty,
            "visual_mean_delta": visual_nll_result["mean_delta"],
            "visual_mean_absolute_delta": visual_nll_result["mean_absolute_delta"],
            "visual_std_delta": visual_nll_result["std_delta"],
            "visual_semantic_log_volume": visual_volume["log_volume"],
        },
        "visual_sensitive_indices": visual_indices.cpu().tolist(),
        "visual_perturbed_nlls": visual_nlls,
        "visual_semantic_volume": visual_volume,
        "metadata": sample["metadata"],
        "runtime_seconds": time.perf_counter() - started,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["cvbench", "cvbench2d", "mmvet", "vilp", "hallusionbench"],
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--perturbation-mode",
        choices=PERTURBATION_MODES,
        default="replace",
    )
    parser.add_argument(
        "--norm-isotropic-sigma",
        type=float,
        default=NORM_ISOTROPIC_SIGMA,
    )
    args = parser.parse_args()
    if not math.isfinite(args.norm_isotropic_sigma) or args.norm_isotropic_sigma <= 0.0:
        parser.error("--norm-isotropic-sigma must be finite and positive")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.manual_seed(SEED)
    done = (
        completed_ids(
            args.output,
            perturbation_mode=args.perturbation_mode,
            norm_isotropic_sigma=args.norm_isotropic_sigma,
        )
        if args.resume
        else set()
    )
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "start dataset=%s completed=%d output=%s", args.dataset, len(done), args.output
    )
    LOGGER.info("loading target_model=LLaVA-1.5-7B")
    llava = LlavaRunner()
    processed = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for sample in iter_samples(args.dataset, limit=args.limit):
            if sample["id"] in done:
                continue
            record = process_one(
                sample,
                llava,
                perturbation_mode=args.perturbation_mode,
                norm_isotropic_sigma=args.norm_isotropic_sigma,
            )
            append_jsonl(handle, record)
            done.add(sample["id"])
            processed += 1
            LOGGER.info(
                "processed=%d id=%s prediction=%r correct=%s uncertainty=%.6f seconds=%.3f peak_gb=%.3f",
                processed,
                record["id"],
                record["prediction"],
                record["correct"],
                record["scores"]["uncertainty"],
                record["runtime_seconds"],
                record["peak_memory_gb"],
            )
    LOGGER.info("complete dataset=%s new_records=%d", args.dataset, processed)


if __name__ == "__main__":
    main()
