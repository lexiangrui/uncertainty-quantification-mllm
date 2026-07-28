"""Utilities for post-projector visual-embedding radius sweeps."""

from __future__ import annotations

import math
import re
from statistics import median


def validate_sigmas(sigmas: list[float]) -> tuple[float, ...]:
    """Return a strictly increasing, positive perturbation grid."""
    values = tuple(float(value) for value in sigmas)
    if not values:
        raise ValueError("at least one sigma is required")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("sigmas must be finite and positive; sigma=0 is measured separately")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("sigmas must be strictly increasing")
    return values


def normalized_answer(text: str) -> str:
    """Normalize only superficial formatting when deciding whether an answer flipped."""
    return re.sub(r"\s+", " ", text.strip()).casefold()


def normalized_trapezoid_auc(points: list[tuple[float, float]], max_sigma: float) -> float:
    """Area under a sigma curve, divided by the scanned sigma range."""
    if max_sigma <= 0:
        raise ValueError("max_sigma must be positive")
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area / max_sigma


def summarize_trials(
    trials: list[dict], sigmas: tuple[float, ...], seeds: tuple[int, ...]
) -> dict:
    """Build per-radius curves and decision-boundary uncertainty summaries."""
    expected = {(sigma, seed) for sigma in sigmas for seed in seeds}
    observed = {(float(item["sigma"]), int(item["seed"])) for item in trials}
    if observed != expected or len(trials) != len(expected):
        raise ValueError("trials must contain exactly one result for every (sigma, seed)")

    curve = []
    for sigma in sigmas:
        items = [item for item in trials if float(item["sigma"]) == sigma]
        curve.append(
            {
                "sigma": sigma,
                "flip_rate": sum(bool(item["answer_changed"]) for item in items) / len(items),
                "mean_delta_nll": sum(float(item["delta_nll"]) for item in items) / len(items),
                "mean_abs_delta_nll": sum(abs(float(item["delta_nll"])) for item in items)
                / len(items),
                "mean_kl": sum(float(item["kl"]) for item in items) / len(items),
                "mean_relative_l2": sum(
                    float(item["geometry"]["mean_token_relative_l2"]) for item in items
                )
                / len(items),
                "mean_cosine": sum(
                    float(item["geometry"]["mean_token_cosine"]) for item in items
                )
                / len(items),
            }
        )

    first_flips = []
    censored = []
    for seed in seeds:
        seed_items = sorted(
            (item for item in trials if int(item["seed"]) == seed),
            key=lambda item: float(item["sigma"]),
        )
        flipped = next((item for item in seed_items if item["answer_changed"]), None)
        if flipped is None:
            censored.append(seed)
        else:
            first_flips.append(
                {
                    "seed": seed,
                    "sigma": float(flipped["sigma"]),
                    "relative_l2": float(flipped["geometry"]["mean_token_relative_l2"]),
                }
            )

    max_sigma = sigmas[-1]

    def with_origin(key: str) -> list[tuple[float, float]]:
        return [(0.0, 0.0)] + [
            (float(item["sigma"]), float(item[key])) for item in curve
        ]

    median_boundary_sigma = next(
        (
            sigma
            for sigma in sigmas
            if sum(item["sigma"] <= sigma for item in first_flips) / len(seeds) >= 0.5
        ),
        None,
    )
    return {
        "curve": curve,
        "first_flip_by_seed": first_flips,
        "no_flip_through_max_sigma_seeds": censored,
        "first_flip_rate": len(first_flips) / len(seeds),
        # Censored directions count in the denominator. If fewer than half of
        # the rays flip, the median boundary lies beyond the scanned range.
        "median_first_flip_sigma": median_boundary_sigma,
        "median_observed_first_flip_sigma": (
            median(item["sigma"] for item in first_flips) if first_flips else None
        ),
        "median_observed_first_flip_relative_l2": (
            median(item["relative_l2"] for item in first_flips) if first_flips else None
        ),
        # Higher AUC means that perturbations destabilize the answer/distribution
        # earlier and over more of the scanned local neighborhood.
        "flip_auc": normalized_trapezoid_auc(with_origin("flip_rate"), max_sigma),
        "kl_auc": normalized_trapezoid_auc(with_origin("mean_kl"), max_sigma),
        "abs_delta_nll_auc": normalized_trapezoid_auc(
            with_origin("mean_abs_delta_nll"), max_sigma
        ),
    }
