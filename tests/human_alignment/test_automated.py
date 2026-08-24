from __future__ import annotations

import json
from pathlib import Path

from src.human_alignment.automated import run_automated_adjudication
from src.human_alignment.workflow import load_annotations, save_annotations


class _Result:
    def to_dict(self):
        return {
            "analysis": "third-judge analysis",
            "correct": False,
            "rating": 1,
            "hallucination": True,
            "hallucination_types": ["reasoning_hallucination"],
            "raw_response": "raw",
        }


class _Judge:
    model = "claude-opus-5"

    def judge(self, **_kwargs):
        return _Result()


def _workspace(path: Path) -> Path:
    workspace = path / "human_alignment"
    workspace.mkdir()
    (workspace / "samples.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "samples": [
                    {
                        "key": "llava/vilp/a",
                        "dataset": "vilp",
                        "question": "question",
                        "references": ["answer"],
                        "response": {"vision": "v", "reasoning": "r", "answer": "a"},
                        "disagreements": {"correct": True, "hallucination": True},
                        "image": None,
                        "image_status": "not_applicable",
                        "content_sha256": "sample-hash",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    save_annotations(
        workspace,
        {
            "llava/vilp/a": {
                "correct": True,
                "provenance": {"correct": {"kind": "human"}},
            }
        },
    )
    return workspace


def test_automated_adjudication_fills_only_missing_fields_and_resumes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    output = workspace / "judging_claude_opus_5.jsonl"
    summary = run_automated_adjudication(
        judge=_Judge(), workspace=workspace, output=output, concurrency=1
    )
    assert summary["succeeded"] == 1
    annotation = load_annotations(workspace)["annotations"]["llava/vilp/a"]
    assert annotation["correct"] is True
    assert annotation["provenance"]["correct"]["kind"] == "human"
    assert annotation["hallucination"] is True
    assert annotation["provenance"]["hallucination"] == {
        "kind": "automated",
        "model": "claude-opus-5",
        "updated_at": annotation["provenance"]["hallucination"]["updated_at"],
    }

    resumed = run_automated_adjudication(
        judge=_Judge(), workspace=workspace, output=output, concurrency=1
    )
    assert resumed["pending"] == 0
    assert resumed["already_completed"] == 1
