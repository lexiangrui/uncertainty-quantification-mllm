"""SAI score computation from extraction records (frozen post-processing).

The extraction persists the full intervention response matrix; every score
here is a pure function of those records, following the LSB freeze
protocol: candidates are compared on the development model only, the final
formula is frozen, then applied unchanged to the other models.

Frozen formula (development model LLaVA-1.5-7B, full 400-sample LUH subset):

    SAI(q) = [z(P31_unembed) + z(P31_mention_state)] / 2 + z(neg_support)

where, per object o of the model's own greedy response (first mention):

* ``P31_<mode>(o)`` — **anchor propagation response**: with the top-16
  lens-located visual states rotated at the mid-depth layer entry by σ=1
  toward (+) / away (−) the object's semantic anchor, the difference of the
  object's vision-logit-lens reading at the last decoder entry,
  (Δreading+ − Δreading−)/2.  Averaged over objects, z-scored within the
  model population, averaged over the two anchor modes (vocabulary
  unembedding / first-mention contextual state).
* ``neg_support(o)`` — minus the baseline teacher-forced logit of the
  object's first mention token; a belief asserted without internal support.

Higher SAI = higher hallucination risk / uncertainty.  The random-direction
control of P31 is chance-level (AUROC 0.513, corr 0.02), so the signal is
anchor-specific rather than generic perturbation damage.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

# Frozen extraction coordinates of the score (LLaVA absolute layer; for
# other depths use the same *fractional* depth: intervene at layer n/2,
# read propagation at the last decoder entry).
FROZEN = {
    "anchor_modes": ("unembed", "mention_state"),
    "locate": "topk",
    "sigma": 1.0,
    "intervene_layer_fraction": 0.5,
    "read_layer": "last_entry",
    "k_locate": 16,
}


@dataclass
class SaiObjectView:
    surface: str
    first_dlogit: dict[tuple, float | None]   # (anchor_mode, locate, layer, sigma, sign) → Δlogit@first mention
    baseline_logit: float | None              # support at first mention
    baseline_logprob: float | None


def parse_record(sai: dict) -> list[SaiObjectView]:
    """Rebuild per-object first-mention response views from one record."""
    objects = sai.get("objects", [])
    base_logit = sai.get("baseline", {}).get("mention_logit")
    base_logprob = sai.get("baseline", {}).get("mention_logprob")
    views = [
        SaiObjectView(
            surface=o["surface"],
            first_dlogit={},
            baseline_logit=(base_logit[i][0] if base_logit and i < len(base_logit) and base_logit[i] else None),
            baseline_logprob=(base_logprob[i][0] if base_logprob and i < len(base_logprob) and base_logprob[i] else None),
        )
        for i, o in enumerate(objects)
    ]
    for iv in sai.get("interventions", []):
        i = next((k for k, o in enumerate(objects) if o["surface"] == iv["anchor"]), None)
        if i is None or iv.get("anchor_mode") == "random":
            continue
        key = (iv["anchor_mode"], iv["locate"], iv["layer"], iv["sigma"], iv["sign"])
        dl = iv.get("mention_dlogit")
        if dl and i < len(dl) and dl[i]:
            v = dl[i][0]
            views[i].first_dlogit[key] = v if v is not None and math.isfinite(v) else None
    return views


def coupling(views: list[SaiObjectView], anchor_mode: str, locate: str, layer: int,
             sigma: float) -> float | None:
    """Mean over objects of C(o) = (Δ_toward − Δ_away)/2 at first mention.

    Higher coupling = the belief in the object strengthens more when the
    visual evidence is pushed toward the object's own semantics.
    """
    vals = []
    for v in views:
        tw = v.first_dlogit.get((anchor_mode, locate, layer, sigma, 1))
        aw = v.first_dlogit.get((anchor_mode, locate, layer, sigma, -1))
        if tw is None or aw is None:
            continue
        vals.append((tw - aw) / 2)
    if not vals:
        return None
    return sum(vals) / len(vals)


def support(views: list[SaiObjectView]) -> float | None:
    """Mean baseline first-mention logit (higher = better supported)."""
    vals = [v.baseline_logit for v in views if v.baseline_logit is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def propagation(sai: dict, anchor_mode: str, locate: str, layer: int, sigma: float,
                read_layer: int) -> float | None:
    """Mean over objects of the reading-propagation response at one layer.

    P(o) = (Δreading_toward − Δreading_away)/2 at ``read_layer`` for the
    object's own anchor interventions.
    """
    objects = sai.get("objects", [])
    vals = []
    for iv in sai.get("interventions", []):
        if (iv.get("anchor_mode") != anchor_mode or iv.get("locate") != locate
                or iv.get("layer") != layer or iv.get("sigma") != sigma
                or iv.get("anchor_mode") == "random"):
            continue
        i = next((k for k, o in enumerate(objects) if o["surface"] == iv["anchor"]), None)
        if i is None:
            continue
        dread = iv.get("lens_dread", {}).get(str(read_layer))
        if not dread or i >= len(dread) or dread[i] is None:
            continue
        vals.append((i, iv["sign"], dread[i]))
    by_obj: dict[int, dict[int, float]] = defaultdict(dict)
    for i, sign, v in vals:
        by_obj[i][sign] = v
    cs = []
    for i, d in by_obj.items():
        if 1 in d and -1 in d:
            cs.append((d[1] - d[-1]) / 2)
    if not cs:
        return None
    return sum(cs) / len(cs)


def zsum(parts: list[list[tuple[str, float]]]) -> dict[str, float]:
    """Equal-weight z-sum of named components within one model population."""
    import numpy as np

    out: dict[str, float] = {}
    names = [n for row in parts for n, _ in row]
    stacked = {n: np.array([dict(row).get(n, float("nan")) for row in parts]) for n in set(names)}
    z = {}
    for n, arr in stacked.items():
        a = arr[np.isfinite(arr)]
        if len(a) < 2:
            z[n] = np.zeros_like(arr)
            continue
        sd = a.std()
        z[n] = np.where(np.isfinite(arr), (arr - a.mean()) / (sd if sd > 1e-12 else 1.0), 0.0)
    for i, row in enumerate(parts):
        out[str(i)] = float(sum(z[n][i] for n, _ in row))
    return out


def sai_frozen_components(record: dict) -> dict[str, float]:
    """Raw components of the frozen SAI score for one extraction record.

    Returns {'P31_unembed': …, 'P31_mention_state': …, 'neg_support': …}
    or an empty dict when the record yields nothing usable.
    """
    sai = record.get("sai", record)
    out: dict[str, float] = {}
    interventions = sai.get("interventions", [])
    if not interventions:
        return out
    layers = sorted({iv.get("layer") for iv in interventions if iv.get("layer") is not None})
    read_layers = set()
    for iv in interventions:
        read_layers.update(int(k) for k in iv.get("lens_dread", {}))
    read_layer = max(read_layers) if read_layers else None
    if read_layer is None:
        return out
    views = parse_record(sai)
    for mode in FROZEN["anchor_modes"]:
        p = propagation(sai, mode, FROZEN["locate"], min(layers), FROZEN["sigma"], read_layer)
        if p is not None:
            out[f"P31_{mode}"] = p
    s = support(views)
    if s is not None:
        out["neg_support"] = -s
    return out


def sai_frozen_scores(components_by_sample: dict[str, dict[str, float]]) -> dict[str, float]:
    """z-combine the frozen components over a model population.

    SAI = [z(P31_unembed) + z(P31_mention_state)]/2 + z(neg_support);
    samples missing a component contribute 0 for its z-term (the remaining
    terms still carry information; coverage is reported alongside).
    """
    import numpy as np

    if not components_by_sample:
        return {}
    sids = sorted(components_by_sample)
    names = ("P31_unembed", "P31_mention_state", "neg_support")
    z: dict[str, np.ndarray] = {}
    for n in names:
        arr = np.array(
            [components_by_sample[s].get(n, np.nan) for s in sids], dtype=float
        )
        finite = arr[np.isfinite(arr)]
        if len(finite) < 2:
            z[n] = np.zeros_like(arr)
        else:
            sd = finite.std()
            z[n] = np.where(np.isfinite(arr), (arr - finite.mean()) / (sd if sd > 1e-12 else 1.0), 0.0)
    out = {}
    for i, sid in enumerate(sids):
        out[sid] = float((z["P31_unembed"][i] + z["P31_mention_state"][i]) / 2 + z["neg_support"][i])
    return out
