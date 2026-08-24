from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from src.llm_judge.closed_source import JUDGE_PROMPT_SHA256

from .workflow import _sha256, load_annotations, save_annotations


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_config(workspace: Path, judge: Any) -> dict[str, Any]:
    return {
        "protocol": "automated-third-judge-v1",
        "judge_model": judge.model,
        "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
        "max_tokens": getattr(judge, "max_tokens", None),
        "samples_input": str((workspace / "samples.json").resolve()),
        "samples_sha256": _sha256(workspace / "samples.json"),
    }


def _load_completed(path: Path, run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if line_number == 1:
                if row != {"record_type": "run", "run": run}:
                    raise ValueError(f"automated adjudication run mismatch: {path}")
                continue
            if row.get("record_type") != "sample" or not isinstance(row.get("key"), str):
                raise ValueError(f"invalid automated adjudication row at {path}:{line_number}")
            if row.get("status") == "ok":
                if not isinstance(row.get("result"), dict):
                    raise ValueError(f"missing automated result at {path}:{line_number}")
                completed[row["key"]] = row
    return completed


def _append_audit(path: Path, run: dict[str, Any], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if path.stat().st_size == 0:
            handle.write(
                json.dumps(
                    {"record_type": "run", "run": run},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.write(
            json.dumps(
                {"record_type": "sample", **row},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _needs_adjudication(sample: dict[str, Any], annotation: dict[str, Any]) -> bool:
    return any(
        sample["disagreements"][field] and type(annotation.get(field)) is not bool
        for field in ("correct", "hallucination")
    )


def _judge_one(judge: Any, workspace: Path, sample: dict[str, Any]) -> dict[str, Any]:
    image = None
    if sample.get("image_status") == "available":
        target = (workspace / sample["image"]).resolve()
        if workspace.resolve() not in target.parents or not target.is_file():
            raise FileNotFoundError(f"alignment image is unavailable: {sample['key']}")
        with Image.open(target) as opened:
            image = opened.convert("RGB").copy()
    elif sample.get("image_status") != "not_applicable":
        raise ValueError(f"alignment image is not ready: {sample['key']}")

    error: Exception | None = None
    for attempt in range(3):
        try:
            result = judge.judge(
                dataset=sample["dataset"],
                question=sample["question"],
                references=list(sample["references"]),
                vision=sample["response"]["vision"],
                reasoning=sample["response"]["reasoning"],
                answer=sample["response"]["answer"],
                image=image,
            )
            return {"status": "ok", "result": result.to_dict(), "error": None}
        except Exception as caught:  # noqa: BLE001 - relay/API errors are retryable
            error = caught
            if attempt < 2:
                time.sleep(15)
    return {
        "status": "error",
        "result": None,
        "error": f"{type(error).__name__}: {error}",
    }


def _apply_result(
    annotation: dict[str, Any],
    sample: dict[str, Any],
    result: dict[str, Any],
    judge_model: str,
) -> dict[str, Any]:
    value = dict(annotation)
    provenance = dict(value.get("provenance") or {})
    timestamp = _now()
    for field in ("correct", "hallucination"):
        if not sample["disagreements"][field] or type(value.get(field)) is bool:
            continue
        value[field] = result[field]
        provenance[field] = {
            "kind": "automated",
            "model": judge_model,
            "updated_at": timestamp,
        }
    if sample["disagreements"]["hallucination"] and result["hallucination"]:
        value["hallucination_types"] = result.get("hallucination_types", [])
    elif sample["disagreements"]["hallucination"]:
        value["hallucination_types"] = []
    value["provenance"] = provenance
    value["annotator"] = value.get("annotator") or f"automated:{judge_model}"
    value["updated_at"] = timestamp
    return value


def run_automated_adjudication(
    *,
    judge: Any,
    workspace: Path,
    output: Path,
    concurrency: int = 10,
    limit: int | None = None,
    checkpoint_every: int = 10,
) -> dict[str, int]:
    if concurrency <= 0 or checkpoint_every <= 0:
        raise ValueError("concurrency and checkpoint_every must be positive")
    queue = json.loads((workspace / "samples.json").read_text(encoding="utf-8"))
    run = _run_config(workspace, judge)
    completed = _load_completed(output, run)
    annotations = load_annotations(workspace)["annotations"]
    queue_by_key = {sample["key"]: sample for sample in queue["samples"]}
    reconciled = 0
    for key, audit in completed.items():
        sample = queue_by_key.get(key)
        if sample is None or audit.get("content_sha256") != sample.get("content_sha256"):
            raise ValueError(f"automated audit sample mismatch: {key}")
        if _needs_adjudication(sample, annotations.get(key, {})):
            annotations[key] = _apply_result(
                annotations.get(key, {}), sample, audit["result"], judge.model
            )
            reconciled += 1
    if reconciled:
        save_annotations(workspace, annotations)
    pending = [
        sample
        for sample in queue["samples"]
        if sample["key"] not in completed
        and _needs_adjudication(sample, annotations.get(sample["key"], {}))
    ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        pending = pending[:limit]

    succeeded = 0
    failed = 0
    dirty = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_judge_one, judge, workspace, sample): sample for sample in pending
        }
        for future in as_completed(futures):
            sample = futures[future]
            try:
                outcome = future.result()
            except Exception as error:  # image/config errors should be audited too
                outcome = {
                    "status": "error",
                    "result": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            _append_audit(
                output,
                run,
                {
                    "key": sample["key"],
                    "content_sha256": sample["content_sha256"],
                    "status": outcome["status"],
                    "result": outcome["result"],
                    "error": outcome["error"],
                    "updated_at": _now(),
                },
            )
            if outcome["status"] == "ok":
                annotations[sample["key"]] = _apply_result(
                    annotations.get(sample["key"], {}),
                    sample,
                    outcome["result"],
                    judge.model,
                )
                succeeded += 1
                dirty += 1
                if dirty >= checkpoint_every:
                    save_annotations(workspace, annotations)
                    dirty = 0
            else:
                failed += 1
    if dirty:
        save_annotations(workspace, annotations)
    return {
        "pending": len(pending),
        "succeeded": succeeded,
        "failed": failed,
        "already_completed": len(completed),
        "reconciled": reconciled,
    }
