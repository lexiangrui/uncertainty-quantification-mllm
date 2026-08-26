from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import binomtest

from src.datasets import iter_dataset
from src.utils import load_jsonl_records


DATASETS = ("vilp", "hallusionbench", "mmvet")
MODELS = ("llava", "qwen", "internvl")
CONDITIONS = ("xml_lora", "native_prompt")
MANIFEST_PROTOCOL = "xml-format-ablation-v2"


def dataset_sources(data_root: Path) -> dict[str, Path]:
    return {
        "vilp": data_root / "vilp/ViLP.parquet",
        "hallusionbench": data_root / "HallusionBench/data",
        "mmvet": data_root / "MMVet/data/test-00000-of-00001.parquet",
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _manifest_hash(samples: list[dict[str, str]]) -> str:
    return hashlib.sha256(_stable_json(samples).encode("utf-8")).hexdigest()


def _complete_xml_ids(path: Path) -> set[str]:
    rows = load_jsonl_records(path)
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError(f"XML generation input lacks a run header: {path}")
    complete: set[str] = set()
    seen: set[str] = set()
    for row in rows[1:]:
        sample_id = row.get("sample", {}).get("sample_id")
        if not isinstance(sample_id, str) or sample_id in seen:
            raise ValueError(f"invalid or duplicate XML generation record in {path}")
        seen.add(sample_id)
        if row.get("greedy", {}).get("sections_valid") is True:
            complete.add(sample_id)
    return complete


def complete_xml_sampling_frame(
    production_generation_root: Path,
) -> tuple[dict[str, set[str]], dict[str, dict[str, int]]]:
    eligible: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        per_model = {
            model: _complete_xml_ids(
                production_generation_root / model / "greedy" / f"{dataset}.jsonl"
            )
            for model in MODELS
        }
        eligible[dataset] = set.intersection(*(set(ids) for ids in per_model.values()))
        counts[dataset] = {
            **{model: len(per_model[model]) for model in MODELS},
            "shared_complete": len(eligible[dataset]),
        }
    return eligible, counts


def prepare_manifest(
    *,
    data_root: Path,
    production_generation_root: Path,
    output: Path,
    sample_size: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    sources = dataset_sources(data_root)
    eligible_by_dataset, xml_complete_counts = complete_xml_sampling_frame(
        production_generation_root
    )
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    source_count = 0
    for dataset in DATASETS:
        for sample in iter_dataset(dataset, sources[dataset]):
            source_count += 1
            if sample.sample_id in seen:
                raise ValueError(f"duplicate sample_id across datasets: {sample.sample_id}")
            seen.add(sample.sample_id)
            if sample.sample_id not in eligible_by_dataset[dataset]:
                continue
            candidates.append(
                {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "dataset": dataset,
                }
            )
    candidates.sort(key=lambda item: (item["dataset"], item["sample_id"]))
    if sample_size > len(candidates):
        raise ValueError(
            f"sample_size={sample_size} exceeds candidate count={len(candidates)}"
        )
    rng = np.random.default_rng(seed)
    indices = sorted(int(value) for value in rng.choice(len(candidates), sample_size, replace=False))
    samples = [candidates[index] for index in indices]
    counts = Counter(item["dataset"] for item in samples)
    manifest = {
        "protocol": MANIFEST_PROTOCOL,
        "sampling_frame": (
            "pooled question instances from ViLP, HallusionBench, and MM-Vet "
            "with complete XML outputs for every tested model"
        ),
        "eligibility_rule": "sections_valid=true for LLaVA, Qwen, and InternVL XML outputs",
        "sampling_method": "simple random sampling without replacement",
        "shared_across_models_and_conditions": True,
        "seed": seed,
        "sample_size": sample_size,
        "pre_filter_candidate_count": source_count,
        "candidate_count": len(candidates),
        "excluded_incomplete_xml_count": source_count - len(candidates),
        "xml_complete_counts": xml_complete_counts,
        "dataset_counts": {dataset: counts.get(dataset, 0) for dataset in DATASETS},
        "sample_set_sha256": _manifest_hash(samples),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError(
                f"refusing to replace an existing, different sample manifest: {output}"
            )
        return existing
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("protocol") != MANIFEST_PROTOCOL:
        raise ValueError(f"unsupported manifest protocol: {value.get('protocol')!r}")
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) != value.get("sample_size"):
        raise ValueError("manifest sample list does not match sample_size")
    if _manifest_hash(samples) != value.get("sample_set_sha256"):
        raise ValueError("manifest sample_set_sha256 mismatch")
    ids = [item.get("sample_id") for item in samples]
    if any(not isinstance(item, str) for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("manifest contains invalid or duplicate sample_id values")
    return value


def selected_ids(manifest: dict[str, Any], dataset: str) -> set[str]:
    return {
        item["sample_id"]
        for item in manifest["samples"]
        if item.get("dataset") == dataset
    }


def materialize_generation_subset(
    *, source: Path, output: Path, sample_ids: set[str], manifest: dict[str, Any]
) -> int:
    rows = load_jsonl_records(source)
    if not rows:
        raise ValueError(f"empty generation input: {source}")
    header = rows[0]
    if header.get("record_type") != "run":
        raise ValueError(f"generation input lacks a run header: {source}")
    selected: list[dict[str, Any]] = []
    found: set[str] = set()
    for row in rows[1:]:
        sample_id = row.get("sample", {}).get("sample_id")
        if sample_id in sample_ids:
            if sample_id in found:
                raise ValueError(f"duplicate generation record: {sample_id}")
            found.add(sample_id)
            selected.append(row)
    missing = sample_ids - found
    if missing:
        raise ValueError(f"generation input is missing selected samples: {sorted(missing)[:5]}")
    run = dict(header.get("run") or {})
    run["ablation_subset"] = {
        "protocol": MANIFEST_PROTOCOL,
        "sample_set_sha256": manifest["sample_set_sha256"],
        "condition": "xml_lora",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(_stable_json({"record_type": "run", "run": run}) + "\n")
        for row in selected:
            handle.write(_stable_json(row) + "\n")
    return len(selected)


def materialize_existing_judge_subset(
    *,
    source: Path,
    output: Path,
    sample_ids: set[str],
    manifest: dict[str, Any],
    judge_model: str,
) -> int:
    rows = load_jsonl_records(source)
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError(f"existing judge input lacks a run header: {source}")
    source_run = rows[0].get("run") or {}
    if (
        source_run.get("protocol") != "openai-responses"
        or source_run.get("judge_model") != judge_model
    ):
        raise ValueError(f"existing judge input has unexpected identity: {source}")
    selected: list[dict[str, Any]] = []
    found: set[str] = set()
    for row in rows[1:]:
        sample_id = row.get("sample", {}).get("sample_id")
        if sample_id not in sample_ids:
            continue
        if sample_id in found:
            raise ValueError(f"duplicate existing judge record: {sample_id}")
        judge = row.get("judge", {})
        if (
            judge.get("valid") is not True
            or type(judge.get("correct")) is not bool
            or type(judge.get("hallucination")) is not bool
        ):
            raise ValueError(f"selected sample lacks a valid existing judge label: {sample_id}")
        found.add(sample_id)
        selected.append(row)
    missing = sample_ids - found
    if missing:
        raise ValueError(f"existing judge input is missing selected samples: {sorted(missing)[:5]}")
    run = dict(source_run)
    run["ablation_subset"] = {
        "protocol": MANIFEST_PROTOCOL,
        "sample_set_sha256": manifest["sample_set_sha256"],
        "condition": "xml_lora",
        "label_source": "existing_single_gemini_judge_labels",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(_stable_json({"record_type": "run", "run": run}) + "\n")
        for row in selected:
            handle.write(_stable_json(row) + "\n")
    return len(selected)


def _records_by_id(path: Path, payload_key: str | None = None) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in load_jsonl_records(path):
        if row.get("record_type") == "run":
            continue
        sample_id = row.get("sample", {}).get("sample_id")
        if not isinstance(sample_id, str) or sample_id in records:
            raise ValueError(f"invalid or duplicate sample record in {path}")
        records[sample_id] = row.get(payload_key, row) if payload_key else row
    return records


def _rate(values: Iterable[bool]) -> float:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        raise ValueError("cannot compute a rate from zero records")
    return float(array.mean())


def _cluster_bootstrap_delta(
    rows: list[dict[str, Any]], metric: str, *, samples: int, seed: int
) -> tuple[float, float]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    keys = sorted(groups)
    if not keys or samples <= 0:
        raise ValueError("bootstrap requires positive samples and non-empty groups")
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for index in range(samples):
        drawn = rng.integers(0, len(keys), size=len(keys))
        boot = [row for value in drawn for row in groups[keys[int(value)]]]
        deltas[index] = _rate(row[f"xml_{metric}"] for row in boot) - _rate(
            row[f"native_{metric}"] for row in boot
        )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return float(low), float(high)


def summarize_model(
    *,
    model: str,
    manifest: dict[str, Any],
    generation_root: Path,
    judge_root: Path,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_samples = {item["sample_id"]: item for item in manifest["samples"]}
    generations: dict[str, dict[str, dict[str, Any]]] = {condition: {} for condition in CONDITIONS}
    judgements: dict[str, dict[str, dict[str, Any]]] = {condition: {} for condition in CONDITIONS}
    for condition in CONDITIONS:
        for dataset in DATASETS:
            generations[condition].update(
                _records_by_id(generation_root / condition / model / f"{dataset}.jsonl")
            )
            judgements[condition].update(
                _records_by_id(judge_root / condition / model / f"{dataset}.jsonl", "judge")
            )
    expected = set(manifest_samples)
    format_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        records = generations[condition]
        if set(records) != expected:
            raise ValueError(f"{model}/{condition}: generation sample set differs from manifest")
        valid = sum(record.get("greedy", {}).get("sections_valid") is True for record in records.values())
        judge_valid = sum(value.get("valid") is True for value in judgements[condition].values())
        format_rows.append(
            {
                "model": model,
                "condition": condition,
                "selected_n": len(expected),
                "format_valid_n": valid,
                "format_valid_rate": valid / len(expected),
                "judge_valid_n": judge_valid,
            }
        )
    paired: list[dict[str, Any]] = []
    for sample_id in sorted(expected):
        native_generation = generations["native_prompt"][sample_id]
        if native_generation.get("greedy", {}).get("sections_valid") is not True:
            continue
        xml = judgements["xml_lora"].get(sample_id, {})
        native = judgements["native_prompt"].get(sample_id, {})
        if xml.get("valid") is not True or native.get("valid") is not True:
            continue
        if any(type(value) is not bool for value in (xml.get("correct"), xml.get("hallucination"), native.get("correct"), native.get("hallucination"))):
            raise ValueError(f"{model}/{sample_id}: valid judge record lacks boolean labels")
        paired.append(
            {
                "sample_id": sample_id,
                "group_id": manifest_samples[sample_id]["group_id"],
                "xml_correct": xml["correct"],
                "native_correct": native["correct"],
                "xml_hallucination": xml["hallucination"],
                "native_hallucination": native["hallucination"],
            }
        )
    if not paired:
        raise ValueError(f"{model}: no paired valid judge records")
    metric_rows: list[dict[str, Any]] = []
    for offset, metric in enumerate(("correct", "hallucination")):
        xml_rate = _rate(row[f"xml_{metric}"] for row in paired)
        native_rate = _rate(row[f"native_{metric}"] for row in paired)
        low, high = _cluster_bootstrap_delta(
            paired,
            metric,
            samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
        )
        xml_only = sum(row[f"xml_{metric}"] and not row[f"native_{metric}"] for row in paired)
        native_only = sum(row[f"native_{metric}"] and not row[f"xml_{metric}"] for row in paired)
        discordant = xml_only + native_only
        p_value = (
            float(binomtest(xml_only, discordant, 0.5).pvalue) if discordant else 1.0
        )
        metric_rows.append(
            {
                "model": model,
                "metric": "accuracy" if metric == "correct" else "hallucination_rate",
                "paired_n": len(paired),
                "candidate_n": len(expected),
                "native_three_part_n": sum(
                    record.get("greedy", {}).get("sections_valid") is True
                    for record in generations["native_prompt"].values()
                ),
                "xml_lora_rate": xml_rate,
                "native_prompt_rate": native_rate,
                "delta_xml_minus_native": xml_rate - native_rate,
                "delta_ci_low": low,
                "delta_ci_high": high,
                "xml_positive_native_negative": xml_only,
                "xml_negative_native_positive": native_only,
                "mcnemar_exact_p": p_value,
            }
        )
    return format_rows, metric_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
