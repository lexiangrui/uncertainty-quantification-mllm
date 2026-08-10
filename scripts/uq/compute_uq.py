#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "semantic_uncertainty_repro" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "perplexity_repro" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "baseline" / "umpire_repro" / "src"))

from perplexity_uq import PerplexityMethod
from sem_unc.semantic_entropy import SemanticEntropyMethod
from src.uq import run_deferred_uq
from umpire_uq import UmpireMethod
from src.llm_judge import NLIJudge


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute UQ from saved generation features.")
    parser.add_argument("--generation-input", required=True, type=Path)
    parser.add_argument("--hidden-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--entailment-model-path", required=True, type=Path)
    parser.add_argument("--entailment-batch-size", type=int, default=32)
    parser.add_argument("--entailment-device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "deferred UQ must run inside a Slurm allocation; submit compute_uq.sbatch"
        )
    entailment = NLIJudge(
        args.entailment_model_path,
        batch_size=args.entailment_batch_size,
        device=args.entailment_device,
    )
    methods = (PerplexityMethod(), SemanticEntropyMethod(entailment), UmpireMethod())
    written, skipped = run_deferred_uq(
        generation_input=args.generation_input,
        hidden_input=args.hidden_input,
        output=args.output,
        methods=methods,
    )
    print(f"completed deferred UQ: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
