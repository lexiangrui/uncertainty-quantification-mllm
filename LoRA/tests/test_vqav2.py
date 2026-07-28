from lora_format.vqav2 import assign_splits, majority_answer, select_distinct_images


def test_majority_answer() -> None:
    annotation = {"answers": [{"answer": "Blue"}] * 8 + [{"answer": "red"}] * 2}
    assert majority_answer(annotation) == ("blue", 8)


def test_selection_uses_distinct_images_and_exact_splits() -> None:
    rows = [
        {"question_type": "color" if index % 2 else "count", "image_id": index, "id": str(index)}
        for index in range(10)
    ]
    selected = select_distinct_images(rows, total=6, seed=42)
    split = assign_splits(selected, train=4, validation=1, test=1)
    assert len({row["image_id"] for row in split}) == 6
    assert [row["split"] for row in split] == ["train"] * 4 + ["validation", "test"]
