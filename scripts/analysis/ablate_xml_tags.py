#!/usr/bin/env python3
"""XML-tag keep-vs-remove ablation for GCAR and ECA.

Tests whether excluding XML tag / whitespace tokens from the self-generated
regions changes the signal, using identical samples and judge labels.

GCAR (from gcar_components_v2, six-region breakdown):
  去除 (final)   s = (scaffold + prefix)          / (total - self)
  保留           s = (scaffold + prefix + other)  / (total - self)
  ("other" = tag/whitespace mass; absorbing it into the numerator equals
   the old contiguous-scaffold definition under piecewise encoding)

ECA (from eca_components_v4, six-bucket masses incl. tags):
  去除 (final)   U_direct = (aAV + aAR)          / (aAI+aAQ+aAV+aAR+aAA)
  保留           U_direct = (aAV + aAR + aTags)  / (aAI+aAQ+aAV+aAR+aAA+aTags)
  (tags are self-generated text; the keep variant treats them as such)

Layer band: GCAR first 4 layers (frozen); ECA layers 0-1 (DIRECT_LAYERS,
frozen).  LUH subsets, in-sample — same caveat as the other ablations.
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
from src.evaluation.metrics import auroc

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("hallusionbench", "vilp", "mmvet")
EPS = 1e-8


def load_labels(model: str, ids: set):
    judge = {}
    for ds in DATASETS:
        p = PROJECT_ROOT / f"results/judging/{model}/{ds}.jsonl"
        if p.exists():
            for obj in load_jsonl_records(p):
                if obj.get("record_type") == "run":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid in ids and obj.get("judge", {}).get("valid") is True:
                    judge[sid] = 1 if obj["judge"]["hallucination"] else 0
    return judge


def band_layers(n_layers: int, band: tuple[float, float]) -> list[int]:
    return [l for l in range(n_layers) if band[0] <= l / (n_layers - 1) <= band[1]]


def gcar_variant(comps: dict, sids: list, layers: list[int], keep: bool) -> np.ndarray:
    out = []
    for s in sids:
        lb = comps[s]["layer_breakdown"]
        num = den = 0.0
        for l in layers:
            r = lb[str(l)]
            num += r[1] + r[2] + (r[5] if keep else 0.0)
            den += sum(r) - r[3]
        out.append(num / (den + EPS))
    return np.array(out)


def eca_direct_variant(comps: dict, sids: list, layers: list[int], keep: bool) -> np.ndarray:
    out = []
    for s in sids:
        e = comps[s]
        heads = e["n_heads"]
        sizes = [e["section_tokens"][k] for k in ("vision", "reasoning", "answer")]
        vals = []
        for l in layers:
            m = e["layer_masses"][str(l)][2]  # answer group
            a = [x / (heads * sizes[2]) for x in m]
            num = a[2] + a[3] + (a[5] if keep else 0.0)
            den = a[0] + a[1] + a[2] + a[3] + a[4] + (a[5] if keep else 0.0)
            vals.append(num / (den + EPS))
        out.append(float(np.mean(vals)))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json")
    parser.add_argument("--gcar-dir", type=Path, default=PROJECT_ROOT / "results/gcar_components_v2")
    parser.add_argument("--eca-dir", type=Path, default=PROJECT_ROOT / "results/eca_components_v4")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/analysis/eca/xml_tag_ablation.json")
    args = parser.parse_args()

    subsets = json.loads(args.subset.read_text(encoding="utf-8"))
    results = {}

    variants = [("去除标签（最终定义）", False), ("保留标签", True)]

    print("== GCAR：prelim 含/不含 XML 标签（前 4 层，分母=全注意力−self）==")
    hdr = f"{'variant':26s}" + "".join(f"{m:>10s}" for m in MODELS)
    print(hdr)
    results["gcar"] = {}
    for name, keep in variants:
        row = f"{name:26s}"
        vals = {}
        for m in MODELS:
            ids = set(subsets[m]["positive_ids"]) | set(subsets[m]["negative_ids"])
            comps, judge = {}, load_labels(m, ids)
            for ds in DATASETS:
                p = args.gcar_dir / f"{m}/{ds}.jsonl"
                if p.exists():
                    for obj in load_jsonl_records(p):
                        if obj.get("record_type") != "sample":
                            continue
                        sid = obj.get("sample", {}).get("sample_id")
                        if sid in ids:
                            comps[sid] = obj["gcar"]
            sids = [s for s in sorted(comps) if s in judge]
            labels = np.array([judge[s] for s in sids], dtype=int)
            L = max(int(k) for k in comps[sids[0]]["layer_breakdown"]) + 1
            v = auroc(gcar_variant(comps, sids, range(min(4, L)), keep), labels)
            vals[m] = v
            row += f"{v:10.4f}"
        results["gcar"][name] = vals
        print(row)

    print("\n== ECA U_direct：自生成项含/不含 XML 标签（层 0–1）==")
    print(hdr)
    results["eca_U_direct"] = {}
    for name, keep in variants:
        row = f"{name:26s}"
        vals = {}
        for m in MODELS:
            ids = set(subsets[m]["positive_ids"]) | set(subsets[m]["negative_ids"])
            comps, judge = {}, load_labels(m, ids)
            for ds in DATASETS:
                p = args.eca_dir / f"{m}/{ds}.jsonl"
                if p.exists():
                    for obj in load_jsonl_records(p):
                        if obj.get("record_type") != "sample":
                            continue
                        sid = obj.get("sample", {}).get("sample_id")
                        if sid in ids:
                            comps[sid] = obj["eca"]
            sids = [s for s in sorted(comps) if s in judge]
            labels = np.array([judge[s] for s in sids], dtype=int)
            v = auroc(eca_direct_variant(comps, sids, [0, 1], keep), labels)
            vals[m] = v
            row += f"{v:10.4f}"
        results["eca_U_direct"][name] = vals
        print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\noutput={args.output}")


if __name__ == "__main__":
    main()
