#!/usr/bin/env python3
"""Layer sweep for ERA (Early Rationale Attribution) on the LUH subsets.

Computes per-layer AUROC for U_ERA vs relative layer depth per model,
plus band-averaged AUROC over fixed relative-depth ranges.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import auroc
from src.improvement.era import FEATURES, layer_features
from src.utils import load_jsonl_records

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("hallusionbench", "vilp", "mmvet")


def load(model: str, ids: set[str], components_dir: Path):
    comps, judge = {}, {}
    for ds in DATASETS:
        p = components_dir / f"{model}/{ds}.jsonl"
        if not p.exists():
            fallback = PROJECT_ROOT / f"results/eca_components/{model}/{ds}.jsonl"
            if fallback.exists():
                p = fallback
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") != "sample":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid in ids:
                    payload = obj.get("era") or obj.get("eca")
                    if payload:
                        comps[sid] = layer_features(payload)
        p_judge = PROJECT_ROOT / f"results/judging/{model}/{ds}.jsonl"
        if p_judge.exists():
            for obj in load_jsonl_records(p_judge):
                if obj.get("record_type") == "run":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid in ids and obj.get("judge", {}).get("valid") is True:
                    judge[sid] = 1 if obj["judge"]["hallucination"] else 0
    missing_comps = ids - comps.keys()
    missing_judge = ids - judge.keys()
    if missing_comps or missing_judge:
        raise ValueError(
            f"{model}: incomplete sweep inputs "
            f"(ERA missing={len(missing_comps)}, judge missing={len(missing_judge)})"
        )
    sids = sorted(ids)
    return sids, np.array([judge[s] for s in sids], dtype=int), comps


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ERA layers across LUH subsets")
    parser.add_argument(
        "--subset",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json",
    )
    parser.add_argument(
        "--components-dir",
        type=Path,
        default=PROJECT_ROOT / "results/era_components",
    )
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/era/layer_sweep.json",
    )
    args = parser.parse_args()

    subsets = json.loads(args.subset.read_text(encoding="utf-8"))
    data = {}
    for m in MODELS:
        sub = subsets[m]
        ids = set(sub["positive_ids"]) | set(sub["negative_ids"])
        data[m] = load(m, ids, args.components_dir)
        print(f"{m}: n={len(data[m][0])}")

    results = {}
    for feat in args.features:
        print(f"\n== {feat}: per-layer AUROC (LUH sweep set) ==")
        header = f"{'layer(rel)':16s}" + "".join(f"{m:>10s}" for m in MODELS)
        print(header)
        results[feat] = {"per_layer": {}}
        Ls = {m: max(data[m][2][data[m][0][0]].keys()) + 1 for m in MODELS}
        Lmax = max(Ls.values())
        for l in range(Lmax):
            row = f"{l:4d}"
            vals = {}
            for m in MODELS:
                sids, labels, comps = data[m]
                if l not in comps[sids[0]]:
                    row += f"{'--':>10s}"
                    continue
                sc = np.array([comps[s][l][feat] for s in sids])
                v = auroc(sc, labels)
                assert v is not None
                vals[m] = v
                row += f"{v:10.4f}"
            print(row)
            results[feat]["per_layer"][str(l)] = vals

        # Band averages over fixed relative-depth ranges.
        print(f"-- {feat}: relative-depth band means --")
        bands = {
            "early [0,1/3)": lambda rel: rel < 1 / 3,
            "mid [1/3,2/3)": lambda rel: 1 / 3 <= rel < 2 / 3,
            "late [2/3,1]": lambda rel: rel >= 2 / 3,
            "mid+late [1/3,1]": lambda rel: rel >= 1 / 3,
            "late-half [1/2,1]": lambda rel: rel >= 1 / 2,
        }
        results[feat]["bands"] = {}
        for bname, keep in bands.items():
            row = f"  {bname:18s}"
            vals = {}
            for m in MODELS:
                sids, labels, comps = data[m]
                L = Ls[m]
                sc = np.array([
                    np.mean([comps[s][l][feat] for l in sorted(comps[s]) if keep(l / (L - 1))])
                    for s in sids
                ])
                v = auroc(sc, labels)
                assert v is not None
                vals[m] = v
                row += f"{v:10.4f}"
            results[feat]["bands"][bname] = vals
            print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\noutput={args.output}")


if __name__ == "__main__":
    main()
