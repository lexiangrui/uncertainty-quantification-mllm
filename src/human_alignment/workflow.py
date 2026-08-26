from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.datasets import iter_dataset


SCHEMA_VERSION = 1
MODELS = ("llava", "qwen", "internvl")
DATASETS = ("vilp", "hallusionbench", "mmvet")
HALLUCINATION_TYPES = {"vision_hallucination", "reasoning_hallucination"}
GPT_JUDGE_MODEL = "gpt-5.6-terra"
GEMINI_JUDGE_MODEL = "gemini-3.7-flash"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_judge_file(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    run: dict[str, Any] | None = None
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if run is None:
                if row.get("record_type") != "run" or not isinstance(row.get("run"), dict):
                    raise ValueError(f"missing run header at {path}:{line_number}")
                run = row["run"]
                continue
            if row.get("record_type") != "sample":
                raise ValueError(f"invalid sample row at {path}:{line_number}")
            sample_id = row.get("sample", {}).get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"invalid sample_id at {path}:{line_number}")
            judge = row.get("judge")
            if not isinstance(judge, dict):
                raise ValueError(f"missing judge result for {sample_id} in {path}")
            # Failed attempts remain in the raw audit trail. Only a later valid
            # verdict participates in alignment; two valid verdicts are ambiguous.
            if judge.get("valid") is not True:
                continue
            if type(judge.get("correct")) is not bool or type(judge.get("hallucination")) is not bool:
                raise ValueError(f"non-boolean judge label for {sample_id} in {path}")
            if sample_id in records:
                raise ValueError(f"duplicate valid sample_id at {path}:{line_number}")
            records[sample_id] = row
    if run is None:
        raise ValueError(f"empty judge file: {path}")
    return run, records


def _same_input(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("sample") == right.get("sample") and left.get("input") == right.get("input")


def _image_name(dataset: str, sample_id: str) -> str:
    suffix = hashlib.sha256(sample_id.encode()).hexdigest()[:12]
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in sample_id)[:80]
    return f"images/{dataset}/{safe}-{suffix}.jpg"


def _export_images(
    samples: list[dict[str, Any]], dataset_sources: dict[str, str], workspace: Path
) -> None:
    by_dataset: dict[str, set[str]] = {}
    for row in samples:
        by_dataset.setdefault(row["dataset"], set()).add(row["sample_id"])
    image_values: dict[tuple[str, str], tuple[str, str | None]] = {}
    for dataset, needed in by_dataset.items():
        source = dataset_sources.get(dataset)
        if not source:
            raise ValueError(f"{dataset}: judge header has no dataset_source; cannot prepare images")
        found: set[str] = set()
        try:
            for sample in iter_dataset(dataset, Path(source)):
                if sample.sample_id not in needed:
                    continue
                found.add(sample.sample_id)
                relative: str | None = None
                if sample.image is not None:
                    relative = _image_name(dataset, sample.sample_id)
                    target = workspace / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image = sample.image
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(target, "JPEG", quality=92)
                status = "available" if relative else "not_applicable"
                image_values[(dataset, sample.sample_id)] = (status, relative)
                if found == needed:
                    break
        except (OSError, ValueError) as error:
            raise ValueError(f"{dataset}: failed to export adjudication images ({error})") from error
        missing = sorted(needed - found)
        if missing:
            raise ValueError(
                f"{dataset}: {len(missing)} alignment samples not found in dataset; "
                f"examples={missing[:3]}"
            )
    for row in samples:
        status, relative = image_values[(row["dataset"], row["sample_id"])]
        row["image_status"] = status
        row["image"] = relative


def load_annotations(workspace: Path) -> dict[str, Any]:
    path = workspace / "annotations.json"
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "updated_at": None, "annotations": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION or not isinstance(value.get("annotations"), dict):
        raise ValueError(f"unsupported annotations file: {path}")
    value["annotations"] = {
        key: _validate_annotation(annotation)
        for key, annotation in value["annotations"].items()
    }
    return value


def _validate_annotation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("annotation must be an object")
    result = {
        "correct": value.get("correct"),
        "hallucination": value.get("hallucination"),
        "hallucination_types": value.get("hallucination_types", []),
        "notes": value.get("notes", ""),
        "annotator": value.get("annotator", ""),
        "updated_at": value.get("updated_at") or _utc_now(),
        "provenance": value.get("provenance", {}),
    }
    if result["correct"] is not None and type(result["correct"]) is not bool:
        raise ValueError("correct must be boolean or null")
    if result["hallucination"] is not None and type(result["hallucination"]) is not bool:
        raise ValueError("hallucination must be boolean or null")
    types = result["hallucination_types"]
    if not isinstance(types, list) or any(item not in HALLUCINATION_TYPES for item in types):
        raise ValueError("invalid hallucination_types")
    result["hallucination_types"] = sorted(set(types))
    if result["hallucination"] is False:
        result["hallucination_types"] = []
    if not isinstance(result["notes"], str) or not isinstance(result["annotator"], str):
        raise ValueError("notes and annotator must be strings")
    provenance = result["provenance"]
    if not isinstance(provenance, dict) or any(
        field not in {"correct", "hallucination"} for field in provenance
    ):
        raise ValueError("invalid annotation provenance")
    normalized_provenance: dict[str, dict[str, Any]] = {}
    for field, entry in provenance.items():
        if not isinstance(entry, dict) or entry.get("kind") != "human":
            raise ValueError(f"invalid {field} provenance")
        model = entry.get("model")
        if model is not None:
            raise ValueError(f"human {field} provenance cannot name a model")
        updated_at = entry.get("updated_at")
        if updated_at is not None and not isinstance(updated_at, str):
            raise ValueError(f"invalid {field} provenance timestamp")
        normalized_provenance[field] = {
            "kind": "human",
            "model": None,
            "updated_at": updated_at,
        }
    result["provenance"] = normalized_provenance
    return result


def save_annotations(workspace: Path, annotations: dict[str, Any]) -> None:
    normalized = {key: _validate_annotation(value) for key, value in annotations.items()}
    _atomic_json(
        workspace / "annotations.json",
        {"schema_version": SCHEMA_VERSION, "updated_at": _utc_now(), "annotations": normalized},
    )


def build_alignment_workspace(
    *,
    gpt_dir: Path,
    gemini_dir: Path,
    workspace: Path,
    models: tuple[str, ...] = MODELS,
    datasets: tuple[str, ...] = DATASETS,
    export_images: bool = True,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    dataset_sources: dict[str, str] = {}
    counts = {"correct": 0, "hallucination": 0, "overlap": 0, "unique": 0}
    for model in models:
        for dataset in datasets:
            gpt_path = gpt_dir / model / f"{dataset}.jsonl"
            gemini_path = gemini_dir / model / f"{dataset}.jsonl"
            gpt_run, gpt_rows = _load_judge_file(gpt_path)
            gemini_run, gemini_rows = _load_judge_file(gemini_path)
            if gpt_run.get("judge_model") != GPT_JUDGE_MODEL:
                raise ValueError(
                    f"unexpected GPT judge_model at {gpt_path}: {gpt_run.get('judge_model')!r}"
                )
            if gemini_run.get("judge_model") != GEMINI_JUDGE_MODEL:
                raise ValueError(
                    f"unexpected Gemini judge_model at {gemini_path}: "
                    f"{gemini_run.get('judge_model')!r}"
                )
            if gpt_run.get("greedy_run") != gemini_run.get("greedy_run"):
                raise ValueError(f"greedy_run mismatch: {model}/{dataset}")
            if set(gpt_rows) != set(gemini_rows):
                missing_gpt = sorted(set(gemini_rows) - set(gpt_rows))[:3]
                missing_gemini = sorted(set(gpt_rows) - set(gemini_rows))[:3]
                raise ValueError(
                    f"sample set mismatch: {model}/{dataset}; missing GPT={missing_gpt}, "
                    f"missing Gemini={missing_gemini}"
                )
            dataset_source = str(gpt_run.get("dataset_source", ""))
            previous_source = dataset_sources.setdefault(dataset, dataset_source)
            if previous_source != dataset_source:
                raise ValueError(
                    f"dataset_source mismatch across models for {dataset}: "
                    f"{previous_source!r} != {dataset_source!r}"
                )
            source_files.extend(
                [
                    {"judge": "gpt", "path": str(gpt_path.resolve()), "sha256": _sha256(gpt_path)},
                    {"judge": "gemini", "path": str(gemini_path.resolve()), "sha256": _sha256(gemini_path)},
                ]
            )
            for sample_id in sorted(gpt_rows):
                gpt = gpt_rows[sample_id]
                gemini = gemini_rows[sample_id]
                if not _same_input(gpt, gemini):
                    raise ValueError(f"sample/input mismatch: {model}/{dataset}/{sample_id}")
                correct = gpt["judge"]["correct"] != gemini["judge"]["correct"]
                hallucination = gpt["judge"]["hallucination"] != gemini["judge"]["hallucination"]
                if not correct and not hallucination:
                    continue
                counts["correct"] += int(correct)
                counts["hallucination"] += int(hallucination)
                counts["overlap"] += int(correct and hallucination)
                counts["unique"] += 1
                value = gpt["input"]
                samples.append(
                    {
                        "key": f"{model}/{dataset}/{sample_id}",
                        "model": model,
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "group_id": gpt["sample"].get("group_id"),
                        "question": value.get("question", ""),
                        "references": value.get("references", []),
                        "response": {
                            "vision": value.get("vision", ""),
                            "reasoning": value.get("reasoning", ""),
                            "answer": value.get("answer", ""),
                        },
                        "disagreements": {"correct": correct, "hallucination": hallucination},
                        "image": None,
                        "image_status": "pending" if export_images else "skipped",
                    }
                )
                samples[-1]["content_sha256"] = hashlib.sha256(
                    json.dumps(
                        {
                            key: value
                            for key, value in samples[-1].items()
                            if key not in {"image", "image_status", "content_sha256"}
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
    if export_images:
        _export_images(samples, dataset_sources, workspace)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "blind": True,
        "sources": source_files,
        "counts": counts,
        "warnings": [],
        "samples": samples,
    }
    existing = load_annotations(workspace)
    old_samples_path = workspace / "samples.json"
    old_samples = {}
    if old_samples_path.is_file():
        old_payload = json.loads(old_samples_path.read_text(encoding="utf-8"))
        old_samples = {row["key"]: row for row in old_payload.get("samples", [])}
    current = {row["key"]: row for row in samples}
    preserved = {
        key: value
        for key, value in existing["annotations"].items()
        if key in current
        and old_samples.get(key, {}).get("content_sha256") == current[key]["content_sha256"]
    }
    _atomic_json(workspace / "samples.json", payload)
    save_annotations(workspace, preserved)
    return payload


def _select_auxiliary_judge(
    gpt: dict[str, Any], gemini: dict[str, Any], hallucination: bool
) -> dict[str, Any]:
    for judge in (gpt["judge"], gemini["judge"]):
        if judge["hallucination"] is hallucination:
            return judge
    return gpt["judge"]


def _write_jsonl(path: Path, run: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "run", "run": run}, ensure_ascii=False, separators=(",", ":")) + "\n")
        for row in rows:
            handle.write(json.dumps({"record_type": "sample", **row}, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def finalize_aligned_results(
    *,
    gpt_dir: Path,
    gemini_dir: Path,
    workspace: Path,
    output_dir: Path,
    models: tuple[str, ...] = MODELS,
    datasets: tuple[str, ...] = DATASETS,
    human_adjudicator: str = "lexiangrui",
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"official output already exists: {output_dir}")
    queue = json.loads((workspace / "samples.json").read_text(encoding="utf-8"))
    annotations = load_annotations(workspace)["annotations"]
    queue_by_key = {row["key"]: row for row in queue["samples"]}
    expected_hashes = {
        (entry["judge"], Path(entry["path"]).resolve()): entry["sha256"]
        for entry in queue.get("sources", [])
    }
    for (_judge_name, raw_path), expected in expected_hashes.items():
        if not raw_path.is_file() or _sha256(raw_path) != expected:
            raise ValueError(f"raw judge input changed after queue preparation: {raw_path}")
    incomplete: list[str] = []
    for key, row in queue_by_key.items():
        annotation = annotations.get(key, {})
        if row["disagreements"]["correct"] and type(annotation.get("correct")) is not bool:
            incomplete.append(f"{key}:correct")
        if row["disagreements"]["hallucination"] and type(annotation.get("hallucination")) is not bool:
            incomplete.append(f"{key}:hallucination")
    if incomplete:
        raise ValueError(f"human alignment is incomplete ({len(incomplete)} fields): {incomplete[:5]}")

    stage = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    file_stats: list[dict[str, Any]] = []
    try:
        for model in models:
            for dataset in datasets:
                gpt_path = gpt_dir / model / f"{dataset}.jsonl"
                gemini_path = gemini_dir / model / f"{dataset}.jsonl"
                for judge_name, raw_path in (("gpt", gpt_path), ("gemini", gemini_path)):
                    expected = expected_hashes.get((judge_name, raw_path.resolve()))
                    if expected is None or _sha256(raw_path) != expected:
                        raise ValueError(f"raw judge input changed after queue preparation: {raw_path}")
                gpt_run, gpt_rows = _load_judge_file(gpt_path)
                gemini_run, gemini_rows = _load_judge_file(gemini_path)
                if set(gpt_rows) != set(gemini_rows) or gpt_run.get("greedy_run") != gemini_run.get("greedy_run"):
                    raise ValueError(f"raw judge inputs changed: {model}/{dataset}")
                final_rows: list[dict[str, Any]] = []
                adjudicated = 0
                for sample_id, gpt in gpt_rows.items():
                    gemini = gemini_rows[sample_id]
                    key = f"{model}/{dataset}/{sample_id}"
                    queue_row = queue_by_key.get(key)
                    correct_disagrees = gpt["judge"]["correct"] != gemini["judge"]["correct"]
                    hallucination_disagrees = gpt["judge"]["hallucination"] != gemini["judge"]["hallucination"]
                    if (correct_disagrees or hallucination_disagrees) and queue_row is None:
                        raise ValueError(f"new disagreement not present in workspace: {key}")
                    annotation = annotations.get(key, {})
                    correct = annotation["correct"] if correct_disagrees else gpt["judge"]["correct"]
                    hallucination = annotation["hallucination"] if hallucination_disagrees else gpt["judge"]["hallucination"]
                    auxiliary = _select_auxiliary_judge(gpt, gemini, hallucination)
                    auxiliary_source = "gpt" if auxiliary is gpt["judge"] else "gemini"
                    judge = dict(auxiliary)
                    judge.update({"status": "ok", "valid": True, "error": None, "correct": correct, "hallucination": hallucination})
                    if hallucination_disagrees:
                        human_types = annotation.get("hallucination_types", [])
                        if hallucination and human_types:
                            judge["hallucination_types"] = human_types
                        elif not hallucination:
                            judge["hallucination_types"] = []
                    correct_source = (
                        {"kind": "human", "model": None}
                        if correct_disagrees
                        else {"kind": "judge_consensus", "model": None}
                    )
                    hallucination_source = (
                        {"kind": "human", "model": None}
                        if hallucination_disagrees
                        else {"kind": "judge_consensus", "model": None}
                    )
                    judge["alignment"] = {
                        "correct": correct_source,
                        "hallucination": hallucination_source,
                        "annotator": (
                            annotation.get("annotator") or human_adjudicator
                            if queue_row
                            else ""
                        ),
                        "updated_at": annotation.get("updated_at") if queue_row else None,
                        "notes": annotation.get("notes", "") if queue_row else "",
                        "auxiliary_fields_source": auxiliary_source,
                    }
                    adjudicated += int(correct_disagrees or hallucination_disagrees)
                    final_rows.append({"sample": gpt["sample"], "input": gpt["input"], "judge": judge})
                run = {
                    "protocol": "human-aligned-dual-judge-v1",
                    "judge_models": [gpt_run.get("judge_model"), gemini_run.get("judge_model")],
                    "human_adjudicator": human_adjudicator,
                    "raw_judge_inputs": [str(gpt_path.resolve()), str(gemini_path.resolve())],
                    "raw_judge_sha256": [_sha256(gpt_path), _sha256(gemini_path)],
                    "annotations_sha256": _sha256(workspace / "annotations.json"),
                    "dataset": dataset,
                    "dataset_source": gpt_run.get("dataset_source"),
                    "greedy_input": gpt_run.get("greedy_input"),
                    "greedy_run": gpt_run.get("greedy_run"),
                }
                _write_jsonl(stage / model / f"{dataset}.jsonl", run, final_rows)
                file_stats.append({"model": model, "dataset": dataset, "samples": len(final_rows), "adjudicated": adjudicated})
        os.replace(stage, output_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    manifest = {"created_at": _utc_now(), "output": str(output_dir.resolve()), "files": file_stats}
    _atomic_json(workspace / "finalization_manifest.json", manifest)
    return manifest
