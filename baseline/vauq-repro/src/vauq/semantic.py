from __future__ import annotations

import math
import re
import string
from collections import Counter
from difflib import SequenceMatcher
from fractions import Fraction


_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation if c not in {".", "/", "-"}})


def normalize_answer(text: object) -> str:
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text):
        try:
            if "/" in match:
                values.append(float(Fraction(match)))
            else:
                values.append(float(match))
        except (ValueError, ZeroDivisionError):
            continue
    return values


def _numeric_close(a: str, b: str, rel_tol: float = 0.03, abs_tol: float = 1e-3) -> bool:
    nums_a = extract_numbers(a)
    nums_b = extract_numbers(b)
    if not nums_a or not nums_b:
        return False
    return any(math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x in nums_a for y in nums_b)


def text_equivalent(a: object, b: object) -> bool:
    na = normalize_answer(a)
    nb = normalize_answer(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if _numeric_close(na, nb):
        return True
    if len(nb) >= 3 and re.search(rf"(?<!\w){re.escape(nb)}(?!\w)", na):
        return True
    if len(na) >= 3 and re.search(rf"(?<!\w){re.escape(na)}(?!\w)", nb):
        return True
    ratio = SequenceMatcher(None, na, nb).ratio()
    if ratio >= 0.88:
        return True
    toks_a = set(na.split())
    toks_b = set(nb.split())
    if not toks_a or not toks_b:
        return False
    jaccard = len(toks_a & toks_b) / len(toks_a | toks_b)
    return jaccard >= 0.82


def cluster_texts(texts: list[str]) -> list[list[int]]:
    clusters: list[list[int]] = []
    reps: list[str] = []
    for idx, text in enumerate(texts):
        placed = False
        for c_idx, rep in enumerate(reps):
            if text_equivalent(text, rep):
                clusters[c_idx].append(idx)
                placed = True
                break
        if not placed:
            reps.append(text)
            clusters.append([idx])
    return clusters


def entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def semantic_entropy(texts: list[str]) -> tuple[float, dict[str, object]]:
    clusters = cluster_texts(texts)
    counts = [len(c) for c in clusters]
    entropy = entropy_from_counts(counts)
    labels: list[int] = [0] * len(texts)
    for c_idx, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = c_idx
    max_prob = max(counts) / len(texts) if texts else 0.0
    return entropy, {
        "num_samples": len(texts),
        "num_clusters": len(clusters),
        "cluster_counts": counts,
        "cluster_labels": labels,
        "cluster_representatives": [texts[c[0]] for c in clusters],
        "max_cluster_probability": max_prob,
        "surface_counts": Counter(normalize_answer(t) for t in texts),
    }
