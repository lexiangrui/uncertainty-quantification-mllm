#!/usr/bin/env python3
"""Analyze correctness versus MMHal-derived hallucination labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmvet", required=True)
    parser.add_argument("--vilp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def read_rows(path: str, dataset: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    valid = []
    for row in rows:
        if row.get("judge_status") != "ok":
            continue
        item = dict(row)
        item["dataset"] = dataset
        item["error"] = not bool(item["correct"])
        item["cluster"] = (
            f"vilp:{int(item['id']) // 2}" if dataset == "vilp"
            else f"mmvet:{item['id']}"
        )
        item["answer_length"] = len(item.get("generated_ids") or [])
        valid.append(item)
    return valid


def metric(rows, label: str) -> dict:
    y = np.asarray([int(row[label]) for row in rows])
    s = np.asarray([float(row["scores"]["vauq"]) for row in rows])
    out = {"n": len(rows), "positive": int(y.sum())}
    if len(rows) == 0 or len(np.unique(y)) < 2:
        return out | {"auroc": float("nan"), "aupr": float("nan")}
    return out | {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),
    }


def clustered_sample(rows, rng):
    groups = {}
    for row in rows:
        groups.setdefault(row["cluster"], []).append(row)
    keys = list(groups)
    sampled = rng.choice(keys, size=len(keys), replace=True)
    return [row for key in sampled for row in groups[key]]


def bootstrap_metrics(rows, label, condition, repeats, rng):
    selected = [row for row in rows if condition(row)]
    point = metric(selected, label)
    values = {"auroc": [], "aupr": []}
    for _ in range(repeats):
        sample = clustered_sample(selected, rng)
        estimate = metric(sample, label)
        for name in values:
            if math.isfinite(estimate[name]):
                values[name].append(estimate[name])
    for name, samples in values.items():
        point[f"{name}_ci95"] = (
            [float(x) for x in np.quantile(samples, [0.025, 0.975])]
            if samples else [float("nan"), float("nan")]
        )
        point[f"{name}_bootstrap_valid"] = len(samples)
    return point


def construct_delta(rows, repeats, rng):
    error = metric(rows, "error")["auroc"]
    hall = metric(rows, "hallucination")["auroc"]
    values = []
    for _ in range(repeats):
        sample = clustered_sample(rows, rng)
        a = metric(sample, "error")["auroc"]
        b = metric(sample, "hallucination")["auroc"]
        if math.isfinite(a) and math.isfinite(b):
            values.append(a - b)
    return {
        "value": error - hall,
        "ci95": [float(x) for x in np.quantile(values, [0.025, 0.975])]
        if values else [float("nan"), float("nan")],
        "bootstrap_valid": len(values),
    }


def quadrant_stats(rows):
    groups = {
        "correct_no_hall": lambda r: r["correct"] and not r["hallucination"],
        "correct_hall": lambda r: r["correct"] and r["hallucination"],
        "error_no_hall": lambda r: r["error"] and not r["hallucination"],
        "error_hall": lambda r: r["error"] and r["hallucination"],
    }
    out = {}
    for name, condition in groups.items():
        scores = np.asarray([row["scores"]["vauq"] for row in rows if condition(row)])
        out[name] = {
            "n": int(len(scores)),
            "mean": float(np.mean(scores)) if len(scores) else float("nan"),
            "median": float(np.median(scores)) if len(scores) else float("nan"),
            "iqr": [float(x) for x in np.quantile(scores, [0.25, 0.75])]
            if len(scores) else [float("nan"), float("nan")],
        }
    return out


def logistic_regression(rows):
    # Hallucination ~ standardized VAUQ + error + log answer length + dataset/case effects.
    vauq = np.asarray([row["scores"]["vauq"] for row in rows], dtype=float)
    vauq = (vauq - vauq.mean()) / (vauq.std(ddof=0) or 1.0)
    length = np.log1p([row["answer_length"] for row in rows])
    dataset = np.asarray([row["dataset"] == "vilp" for row in rows], dtype=float)
    case2 = np.asarray([row.get("subset") == "case2" for row in rows], dtype=float)
    x = np.column_stack([
        np.ones(len(rows)), vauq,
        [row["error"] for row in rows], length, dataset, case2,
    ]).astype(float)
    y = np.asarray([row["hallucination"] for row in rows], dtype=float)
    names = ["intercept", "vauq_z", "error", "log1p_answer_tokens", "dataset_vilp", "vilp_case2"]
    beta = np.zeros(x.shape[1])
    ridge = np.diag([0.0] + [1e-8] * (x.shape[1] - 1))
    for _ in range(100):
        eta = np.clip(x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        hessian = x.T @ (w[:, None] * x) + ridge
        step = np.linalg.solve(hessian, x.T @ (y - p) - ridge @ beta)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    covariance = np.linalg.pinv(hessian)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    z = beta / np.where(se > 0, se, np.nan)
    return {
        name: {
            "coefficient": float(b), "standard_error": float(s),
            "p_value": float(2 * norm.sf(abs(zz))),
            "ci95": [float(b - 1.96 * s), float(b + 1.96 * s)],
        }
        for name, b, s, zz in zip(names, beta, se, z)
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    datasets = {
        "mmvet": read_rows(args.mmvet, "mmvet"),
        "vilp": read_rows(args.vilp, "vilp"),
    }
    result = {"bootstrap_repeats": args.bootstrap, "bootstrap_seed": args.seed, "datasets": {}}
    for name, rows in datasets.items():
        result["datasets"][name] = {
            "n_valid": len(rows),
            "metrics": {
                "error_all": bootstrap_metrics(rows, "error", lambda _: True, args.bootstrap, rng),
                "hallucination_all": bootstrap_metrics(rows, "hallucination", lambda _: True, args.bootstrap, rng),
                "hallucination_given_error": bootstrap_metrics(rows, "hallucination", lambda r: r["error"], args.bootstrap, rng),
                "hallucination_given_correct": bootstrap_metrics(rows, "hallucination", lambda r: r["correct"], args.bootstrap, rng),
            },
            "construct_delta_error_minus_hallucination": construct_delta(rows, args.bootstrap, rng),
            "quadrants": quadrant_stats(rows),
        }
    all_rows = datasets["mmvet"] + datasets["vilp"]
    result["combined_logistic_regression"] = logistic_regression(all_rows)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
