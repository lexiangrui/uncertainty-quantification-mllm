#!/usr/bin/env python3
"""Ablation experiments for EPAR.

Two blocks, both computed offline from the per-layer 5-region breakdown
recorded in results/epar_components (single forward pass per sample):

A. 注意力来源消融 (V1 vs V5): fix EPAR's layer window (first 4 layers) and
   swap only the attention source feeding the score —
     视觉来源  s = -A_vis/total        (VGS/V1 的来源)
     prompt来源 s = +A_txt/total        (方向由 dev 定)
     自生成来源 s = +A_prelim/total     (EPAR/V5 的来源)
   plus the original VGS (visual source, last-2/3 layers) as reference.

B. EPAR 细节消融: prelim region definition (scaffold/prefix/self) and layer
   window (width & position).

Direction convention: higher score = higher hallucination risk; source signs
are fixed on the dev model (llava) and applied unchanged to the others.
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
DEV = "llava"
EPAR_LAYERS = 4


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


# breakdown row: [vis, scaffold, prefix, self, text]
def region_score(comps: dict, sids: list, layers: range | list, idxs: tuple[int, ...], sign: float) -> np.ndarray:
    out = []
    for s in sids:
        lb = comps[s]["layer_breakdown"]
        num = sum(sum(lb[str(l)][i] for i in idxs) for l in layers)
        tot = sum(sum(lb[str(l)]) for l in layers)
        out.append(sign * num / tot if tot > 1e-12 else 0.0)
    return np.array(out)


def load(model: str, subset_ids: set):
    comps, judge = {}, {}
    for ds in DATASETS:
        p = PROJECT_ROOT / f"results/epar_components/{model}/{ds}.jsonl"
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") != "sample":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid in subset_ids:
                    comps[sid] = obj.get("epar") or obj.get("vgs")
        p = PROJECT_ROOT / f"results/judging/{model}/{ds}.jsonl"
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") == "run":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid in subset_ids and obj.get("judge", {}).get("valid") is True:
                    judge[sid] = 1 if obj["judge"]["hallucination"] else 0
    sids = [s for s in sorted(comps) if s in judge]
    return sids, np.array([judge[s] for s in sids], dtype=int), comps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/analysis/luh/epar_ablation.json")
    args = parser.parse_args()

    subsets = json.loads(args.subset.read_text(encoding="utf-8"))
    data = {}
    for m in MODELS:
        sub = subsets[m]
        ids = set(sub["positive_ids"]) | set(sub["negative_ids"])
        data[m] = load(m, ids)

    results = {"source_ablation": {}, "detail_ablation": {"prelim_region": {}, "layer_window_width": {}, "layer_window_position": {}}}

    # ---------------- A. attention-source ablation (V1 vs V5) ----------------
    # signs fixed on dev: visual negative, prelim positive, prompt-text positive
    sources = [
        ("视觉来源 -A_vis/total (V1/VGS)", (0,), -1.0),
        ("自生成来源 A_prelim/total (V5/EPAR)", (1, 2), +1.0),
        ("prompt来源 A_txt/total", (4,), +1.0),
    ]
    print("== A. 注意力来源消融 (层窗口=前4层, 只换来源) ==")
    hdr = f"{'source':38s}" + "".join(f"{m:>10s}" for m in MODELS)
    print(hdr)
    dev_sids, dev_labels, dev_comps = data[DEV]
    rows = {}
    for name, idxs, sign in sources:
        a = auroc(region_score(dev_comps, dev_sids, range(EPAR_LAYERS), idxs, sign), dev_labels)
        fixed_sign = sign if a >= 0.5 else -sign  # direction fixed on dev
        vals = {}
        row = f"{name:38s}"
        for m in MODELS:
            sids, labels, comps = data[m]
            v = auroc(region_score(comps, sids, range(EPAR_LAYERS), idxs, fixed_sign), labels)
            vals[m] = v
            row += f"{v:10.4f}"
        # original VGS reference: visual source over last-2/3 layers
        if idxs == (0,):
            refs = []
            for m in MODELS:
                sids, labels, comps = data[m]
                L = max(int(k) for k in comps[sids[0]]["layer_breakdown"]) + 1
                refs.append(f"{m}={auroc(region_score(comps, sids, range(L // 3, L), (0,), -1.0), labels):.4f}")
            row += "   (VGS 原层段参照: " + " ".join(refs) + ")"
        rows[name] = vals
        print(row)
    results["source_ablation"] = rows

    # ---------------- B1. prelim region definition ----------------
    print("\n== B1. EPAR 细节: prelim 区域定义 (前4层) ==")
    print(hdr)
    variants = [
        ("scaffold+prefix (无self, 最终定义)", (1, 2)),
        ("scaffold+prefix+self (含self)", (1, 2, 3)),
        ("scaffold only (推理脚手架)", (1,)),
        ("prefix only (答案前缀)", (2,)),
        ("self only (诊断)", (3,)),
    ]
    rows = {}
    for name, idxs in variants:
        vals = {}
        row = f"{name:38s}"
        for m in MODELS:
            sids, labels, comps = data[m]
            v = auroc(region_score(comps, sids, range(EPAR_LAYERS), idxs, +1.0), labels)
            vals[m] = v
            row += f"{v:10.4f}"
        rows[name] = vals
        print(row)
    results["detail_ablation"]["prelim_region"] = rows

    # ---------------- B2. layer window width ----------------
    print("\n== B2. EPAR 细节: 层窗口宽度 [0..k] ==")
    print(hdr)
    Lmax = {m: max(int(k) for k in data[m][2][data[m][0][0]]["layer_breakdown"]) + 1 for m in MODELS}
    rows = {}
    for k in list(range(0, 9)) + ["all"]:
        vals = {}
        row = f"  [0..{k}]{'':30s}"[:38]
        for m in MODELS:
            sids, labels, comps = data[m]
            ls = range(Lmax[m]) if k == "all" else range(k + 1)
            v = auroc(region_score(comps, sids, ls, (1, 2), +1.0), labels)
            vals[str(k)] = v
            row += f"{v:10.4f}"
        rows[str(k)] = vals
        print(row)
    results["detail_ablation"]["layer_window_width"] = rows

    # ---------------- B3. layer window position (4-layer sliding) ----------------
    print("\n== B3. EPAR 细节: 4层窗口位置 [j..j+3] ==")
    print(hdr)
    rows = {}
    for j in [0, 2, 4, 8, 12, 20, 24]:
        vals = {}
        row = f"  [{j}..{j+3}]{'':30s}"[:38]
        for m in MODELS:
            sids, labels, comps = data[m]
            if j + 4 > Lmax[m]:
                vals[m] = None
                row += f"{'--':>10s}"
                continue
            v = auroc(region_score(comps, sids, range(j, j + 4), (1, 2), +1.0), labels)
            vals[m] = v
            row += f"{v:10.4f}"
        rows[str(j)] = vals
        print(row)
    results["detail_ablation"]["layer_window_position"] = rows

    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\noutput={args.output}")


if __name__ == "__main__":
    main()
