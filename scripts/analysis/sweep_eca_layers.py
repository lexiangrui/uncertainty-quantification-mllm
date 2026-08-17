#!/usr/bin/env python3
"""Layer sweep for ECA on the LUH subsets.

Per layer (heads averaged, PAS row convention) compute the feature chain
    aAI / U_direct / G_V / G_R / U_ECA
and report AUROC vs relative layer depth per model, plus band-averaged
AUROC over relative-depth thirds and sliding 4-layer windows.

Sweep set: the 400-sample LUH subsets (per project decision, 2026-08-18).
Bands chosen from this sweep are selected on the evaluation set — any
reported number is in-sample for the band choice and must be labelled as
such.
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

from src.utils import load_jsonl_records

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("hallusionbench", "vilp", "mmvet")
EPS = 1e-8

FEATURES = ("aAI", "aAQ", "aAV", "aAR", "U_direct", "G_V", "G_R", "U_ECA")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([neg, pos])
    sorted_idx = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[sorted_idx]
    ranks = np.empty(len(combined))
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[sorted_idx[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_neg = len(neg)
    rank_sum_pos = ranks[n_neg:].sum()
    return float((rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * n_neg))


def layer_features(e: dict) -> dict[int, dict[str, float]]:
    """Per-layer feature chain from one sample's recorded masses."""
    H = e["n_heads"]
    st = e["section_tokens"]
    sizes = [st["vision"], st["reasoning"], st["answer"]]
    out = {}
    for l, m in e["layer_masses"].items():
        a = [[m[g][b] / (H * sizes[g]) for b in range(5)] for g in range(3)]
        aVI, aVQ = a[0][0], a[0][1]
        aRI, aRQ, aRV = a[1][0], a[1][1], a[1][2]
        aAI, aAQ, aAV, aAR = a[2][0], a[2][1], a[2][2], a[2][3]
        G_V = aVI / (aVI + aVQ + EPS)
        G_R = (aRI + aRV * G_V) / (aRI + aRQ + aRV + EPS)
        G_A = (aAI + aAV * G_V + aAR * G_R) / (aAI + aAQ + aAV + aAR + EPS)
        out[int(l)] = {
            "aAI": aAI, "aAQ": aAQ, "aAV": aAV, "aAR": aAR,
            "U_direct": (aAV + aAR) / (aAV + aAR + aAI + aAQ + EPS),
            "G_V": G_V, "G_R": G_R,
            "U_ECA": 1.0 - G_A,
        }
    return out


def load(model: str, ids: set):
    comps, judge = {}, {}
    for ds in DATASETS:
        p = PROJECT_ROOT / f"results/eca_components/{model}/{ds}.jsonl"
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") != "sample":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid in ids:
                    comps[sid] = layer_features(obj["eca"])
        p = PROJECT_ROOT / f"results/judging/{model}/{ds}.jsonl"
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") == "run":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid in ids and obj.get("judge", {}).get("valid") is True:
                    judge[sid] = 1 if obj["judge"]["hallucination"] else 0
    sids = [s for s in sorted(comps) if s in judge]
    return sids, np.array([judge[s] for s in sids], dtype=int), comps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-dir", type=Path, default=PROJECT_ROOT / "results/analysis/eca/luh_ids")
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/analysis/eca/layer_sweep.json")
    args = parser.parse_args()

    data = {}
    for m in MODELS:
        ids = {l.strip() for l in (args.ids_dir / f"{m}.txt").open() if l.strip()}
        data[m] = load(m, ids)
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
                vals[m] = v
                row += f"{v:10.4f}"
            rel = min(l / (Ls[m] - 1) for m in MODELS if m in vals) if vals else 0.0
            row += f"   (rel≈{rel:.2f})"
            print(row)
            results[feat]["per_layer"][str(l)] = vals

        # band averages over relative-depth thirds and last-third quarters
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
                vals[m] = v
                row += f"{v:10.4f}"
            results[feat]["bands"][bname] = vals
            print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\noutput={args.output}")


if __name__ == "__main__":
    main()
