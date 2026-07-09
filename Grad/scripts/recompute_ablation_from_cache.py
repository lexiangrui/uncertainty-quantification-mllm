#!/usr/bin/env python3
"""Recompute Grad-VAUQ ablation scores from cached generated ids and indices.

This is the fast path for comparing ablation baselines. It reuses each row's
`generated_ids` and `grad.selected_indices`, then performs only the masked
teacher-forced forward pass for the requested baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "baseline" / "vauq-repro" / "src"
GRAD_ROOT = ROOT / "Grad"
for path in (SRC, GRAD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grad_vauq.backends import build_backend
from grad_vauq.scoring import compute_response_entropy
from vauq.benchmarks import build_benchmark
from vauq.eval import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Grad-VAUQ JSONL with generated_ids and selected_indices.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["llava"], default="llava")
    parser.add_argument("--benchmark", choices=["cvbench", "mmvet", "vilp"], default="cvbench")
    parser.add_argument("--model-path", default=os.environ.get("LLAVA_7B_HF_MODEL"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--ablation-baseline", choices=["zero", "mean"], default="mean")
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: Path) -> set[str]:
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


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    missing_generated = [row.get("id") for row in rows if "generated_ids" not in row]
    if missing_generated:
        raise SystemExit(
            "Input JSONL does not contain generated_ids. Re-run Grad-VAUQ with "
            "the updated runner first. Example missing id: "
            f"{missing_generated[0]}"
        )

    benchmark = build_benchmark(args.benchmark)
    backend = build_backend(
        args.backend,
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        adapter="llava",
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path) if args.resume else set()
    mode = "a" if args.resume and done else "w"

    records: list[dict] = []
    with out_path.open(mode, encoding="utf-8") as f:
        for row in tqdm(rows, desc=f"Recompute {args.ablation_baseline}"):
            row_id = str(row["id"])
            if row_id in done:
                continue
            sample = benchmark.retrieve(int(row_id))
            generated_ids = torch.tensor([row["generated_ids"]], dtype=torch.long, device=backend.device)
            selected_indices = torch.tensor(row["grad"]["selected_indices"], dtype=torch.long, device=backend.device)

            masked_logits, prompt_len = backend.forward_logits_with_ablation(
                sample["img"],
                sample["question"],
                generated_ids,
                selected_indices=selected_indices,
                baseline=args.ablation_baseline,
            )
            entropy_masked = compute_response_entropy(
                masked_logits, generated_ids, prompt_len, backend.device
            )
            entropy_org = float(row["scores"]["entropy"])
            is_score = entropy_masked - entropy_org
            vauq = entropy_org - args.alpha * is_score

            new_row = dict(row)
            new_row["scores"] = {
                **row["scores"],
                "entropy_masked": entropy_masked,
                "is_score": is_score,
                "vauq": vauq,
            }
            new_row["config"] = {
                **row.get("config", {}),
                "ablation_baseline": args.ablation_baseline,
                "recomputed_from": args.input,
            }
            f.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            f.flush()
            records.append(new_row)

    if args.resume and out_path.exists():
        records = load_rows(out_path)

    summary = summarize(records)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
