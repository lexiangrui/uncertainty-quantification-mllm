#!/usr/bin/env python3
"""Exploratory analysis of SAI dev-slice extraction (Stage A + B patterns)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl_records


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    combined = np.concatenate([neg, pos])
    sorted_idx = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[sorted_idx]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[sorted_idx[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_neg = len(neg)
    return float((ranks[n_neg:].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * n_neg))


def load_dev(model, suffix, subset_path):
    subsets = json.loads(Path(subset_path).read_text())
    pos_ids, neg_ids = set(subsets[model]["positive_ids"]), set(subsets[model]["negative_ids"])
    records = {}
    for ds in ("vilp", "hallusionbench", "mmvet"):
        p = PROJECT_ROOT / f"results/sai/{model}/{suffix}{ds}.jsonl"
        if not p.exists():
            continue
        for obj in load_jsonl_records(p):
            if obj.get("record_type") != "sample":
                continue
            sid = obj["sample"]["sample_id"]
            records[sid] = obj["sai"]
    return records, pos_ids, neg_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llava")
    parser.add_argument("--suffix", default="dev_")
    parser.add_argument("--subset", default=str(PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json"))
    args = parser.parse_args()

    records, pos_ids, neg_ids = load_dev(args.model, args.suffix, args.subset)
    print(f"loaded {len(records)} samples "
          f"(pos={sum(s in pos_ids for s in records)}, neg={sum(s in neg_ids for s in records)})")

    # ---------------- Stage A: static lens reading ----------------
    # Per (sample, object): max lens logprob per layer. Sample-level: mean over objects.
    lens_layers = sorted({e["layer"] for r in records.values() for e in r["lens"]})
    print("\n== Stage A: static lens reading (sample-level mean over objects) AUROC pos=hallucination")
    for L in lens_layers:
        stats = {}
        for sid, r in records.items():
            vals = [e["max_logprob"] for e in r["lens"] if e["layer"] == L]
            vals2 = [e["topk_mean_logprob"] for e in r["lens"] if e["layer"] == L]
            if vals:
                stats[sid] = (np.mean(vals), np.mean(vals2))
        for k, idx in (("max", 0), ("topk_mean", 1)):
            sids = sorted(stats)
            scores = [stats[s][idx] for s in sids]
            labels = [1 if s in pos_ids else 0 for s in sids]
            print(f"  layer {L:2d} {k:9s}: AUROC(-reading)={1 - auroc(scores, labels):.3f}  "
                  f"n={len(sids)}  (reading higher=stronger evidence)")

    # ---------------- Stage B: intervention response ----------------
    intervene_layers = sorted({iv["layer"] for r in records.values() for iv in r["interventions"]})
    sigmas = sorted({iv["sigma"] for r in records.values() for iv in r["interventions"]})

    def obj_index(r, surface):
        for i, o in enumerate(r["objects"]):
            if o["surface"] == surface:
                return i
        return None

    print("\n== Stage B: per-object self-coupling C(o) = mean_p [Δlogp_toward − Δlogp_away]/2")
    print("   (positive = mention probability rises when visual evidence pushed toward object)")
    for L in intervene_layers:
        for sigma in sigmas:
            per_sample = {}
            for sid, r in records.items():
                toward = {}
                away = {}
                for iv in r["interventions"]:
                    if iv["layer"] != L or iv["sigma"] != sigma or iv["anchor"].startswith("rand"):
                        continue
                    key = iv["anchor"]
                    tgt = {key: iv["mention_dlogp"]} if key not in toward else toward[key]
                    if iv["sign"] > 0:
                        toward[key] = iv["mention_dlogp"]
                    else:
                        away[key] = iv["mention_dlogp"]
                couplings = []
                for o in r["objects"]:
                    key = o["surface"]
                    if key not in toward or key not in away:
                        continue
                    tw = np.mean([v for v in toward[key][obj_index(r, key)]])
                    aw = np.mean([v for v in away[key][obj_index(r, key)]])
                    couplings.append((tw - aw) / 2)
                if couplings:
                    per_sample[sid] = (np.mean(couplings), np.min(couplings), np.max(couplings))
            for agg, idx in (("mean", 0), ("min", 1), ("max", 2)):
                sids = sorted(per_sample)
                scores = [per_sample[s][idx] for s in sids]
                labels = [1 if s in pos_ids else 0 for s in sids]
                if len(set(labels)) < 2:
                    continue
                print(f"  layer {L:2d} σ={sigma}: AUROC(-C_{agg})={1 - auroc(scores, labels):.3f} "
                      f"n={len(sids)}  pos_mean={np.mean([per_sample[s][idx] for s in sids if s in pos_ids]):+.4f} "
                      f"neg_mean={np.mean([per_sample[s][idx] for s in sids if s in neg_ids]):+.4f}")

    # random control comparison
    print("\n== Random-control |Δlogp| (magnitude) vs semantic |coupling| at each layer/σ")
    for L in intervene_layers:
        for sigma in sigmas:
            sem_mags, rnd_mags, labels = [], [], []
            for sid, r in records.items():
                sem, rnd = [], []
                for iv in r["interventions"]:
                    if iv["layer"] != L or iv["sigma"] != sigma:
                        continue
                    mag = np.mean([abs(v) for row in iv["mention_dlogp"] for v in row])
                    if iv["anchor"].startswith("rand"):
                        rnd.append(mag)
                    else:
                        i = obj_index(r, iv["anchor"])
                        if i is not None and i < len(iv["mention_dlogp"]):
                            sem.append(mag)
                if sem and rnd:
                    sem_mags.append(np.mean(sem))
                    rnd_mags.append(np.mean(rnd))
                    labels.append(1 if sid in pos_ids else 0)
            if sem_mags and len(set(labels)) == 2:
                print(f"  layer {L:2d} σ={sigma}: |sem|={np.mean(sem_mags):+.4f} |rand|={np.mean(rnd_mags):+.4f} "
                      f"ratio={np.mean(sem_mags) / max(np.mean(rnd_mags), 1e-9):.2f}")

    # section dNLL response
    print("\n== Section ΔNLL under toward-interventions (mean over objects/σ)")
    for L in intervene_layers:
        acc = {"pos": [], "neg": []}
        for sid, r in records.items():
            vals = {"vision": [], "reasoning": [], "answer": []}
            for iv in r["interventions"]:
                if iv["layer"] != L or iv["anchor"].startswith("rand") or iv["sign"] < 0:
                    continue
                for sec in vals:
                    if sec in iv["section_dnll"]:
                        vals[sec].append(iv["section_dnll"][sec])
            key = "pos" if sid in pos_ids else "neg"
            acc[key].append({sec: (np.mean(v) if v else float("nan")) for sec, v in vals.items()})
        for sec in ("vision", "reasoning", "answer"):
            p = [a[sec] for a in acc["pos"] if not np.isnan(a.get(sec, np.nan))]
            n = [a[sec] for a in acc["neg"] if not np.isnan(a.get(sec, np.nan))]
            if p and n:
                print(f"  layer {L:2d} {sec:9s}: pos={np.mean(p):+.4f} neg={np.mean(n):+.4f} "
                      f"gap={np.mean(p) - np.mean(n):+.4f}")


if __name__ == "__main__":
    main()
