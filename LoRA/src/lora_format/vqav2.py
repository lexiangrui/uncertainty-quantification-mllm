from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


QUESTION_FILE = "v2_OpenEnded_mscoco_train2014_questions.json"
ANNOTATION_FILE = "v2_mscoco_train2014_annotations.json"


def read_zip_json(path: Path, filename: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as archive:
        matches = [name for name in archive.namelist() if name.endswith(filename)]
        if len(matches) != 1:
            raise ValueError(f"expected one {filename} in {path}, found {matches}")
        with archive.open(matches[0]) as handle:
            return json.load(handle)


def majority_answer(annotation: dict) -> tuple[str, int]:
    answers = [" ".join(row["answer"].strip().lower().split()) for row in annotation["answers"]]
    answer, agreement = Counter(answers).most_common(1)[0]
    return answer, agreement


def collect_candidates(
    questions_zip: Path,
    annotations_zip: Path,
    min_agreement: int,
) -> list[dict]:
    questions = read_zip_json(questions_zip, QUESTION_FILE)["questions"]
    annotations = read_zip_json(annotations_zip, ANNOTATION_FILE)["annotations"]
    by_question = {row["question_id"]: row for row in annotations}
    rows = []
    for question in questions:
        annotation = by_question.get(question["question_id"])
        if annotation is None:
            continue
        answer, agreement = majority_answer(annotation)
        if agreement < min_agreement or not answer or len(answer) > 80 or "<" in answer or ">" in answer:
            continue
        rows.append(
            {
                "id": f"vqav2-{question['question_id']}",
                "question_id": int(question["question_id"]),
                "image_id": int(question["image_id"]),
                "image_file": f"COCO_train2014_{int(question['image_id']):012d}.jpg",
                "question": " ".join(question["question"].strip().split()),
                "answer": answer,
                "agreement": agreement,
                "question_type": annotation.get("question_type", "unknown"),
                "answer_type": annotation.get("answer_type", "unknown"),
            }
        )
    return rows


def select_distinct_images(rows: list[dict], total: int, seed: int) -> list[dict]:
    """Balance question types while selecting at most one question per image."""
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    queues = []
    for name in sorted(by_type):
        rng.shuffle(by_type[name])
        queues.append(deque(by_type[name]))
    rng.shuffle(queues)
    selected: list[dict] = []
    used_images: set[int] = set()
    while queues and len(selected) < total:
        next_round = []
        for queue in queues:
            while queue and queue[0]["image_id"] in used_images:
                queue.popleft()
            if queue:
                row = queue.popleft()
                selected.append(row)
                used_images.add(row["image_id"])
                if len(selected) == total:
                    break
            if queue:
                next_round.append(queue)
        queues = next_round
    if len(selected) != total:
        raise RuntimeError(f"only {len(selected)} distinct-image candidates; requested {total}")
    return selected


def assign_splits(rows: list[dict], train: int, validation: int, test: int) -> list[dict]:
    if len(rows) != train + validation + test:
        raise ValueError("split sizes must equal selected candidate count")
    boundaries = (train, train + validation)
    output = []
    for index, row in enumerate(rows):
        split = "train" if index < boundaries[0] else "validation" if index < boundaries[1] else "test"
        output.append({**row, "split": split})
    return output


def download_image(
    row: dict,
    image_dir: Path,
    timeout: int = 60,
    image_base_url: str = "http://images.cocodataset.org/train2014",
) -> None:
    destination = image_dir / row["image_file"]
    if destination.is_file() and destination.stat().st_size > 0:
        return
    url = image_base_url.rstrip("/") + "/" + row["image_file"]
    temporary = destination.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=timeout) as source, temporary.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    temporary.replace(destination)


def download_images(
    rows: list[dict],
    image_dir: Path,
    workers: int,
    image_base_url: str = "http://images.cocodataset.org/train2014",
) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(
            executor.map(
                lambda row: download_image(row, image_dir, image_base_url=image_base_url),
                rows,
            )
        )
