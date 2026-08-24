from scripts.extract_intersection_matched_subset import extract_intersection_matched


def _row(sample_id: str, values: tuple[float, float, float], hallucination: bool):
    return {
        "sample_id": sample_id,
        "dataset": "vilp",
        "group_id": sample_id,
        "hallucination": hallucination,
        "scores": dict(zip(("perplexity", "semantic_entropy", "umpire"), values)),
    }


def test_intersection_positives_get_unique_matched_negatives():
    rows = [
        _row("n1", (1, 1, 1), False),
        _row("n2", (2, 2, 2), False),
        _row("n3", (3, 3, 3), False),
        _row("n4", (4, 4, 4), False),
        _row("p1", (1, 1, 1), True),
        _row("p2", (2, 1, 2), True),
        _row("excluded", (1, 3, 1), True),
    ]

    result = extract_intersection_matched(rows, 0.5)

    assert set(result["positive_ids"]) == {"p1", "p2"}
    assert result["n_positive"] == result["n_negative"] == 2
    assert len(set(result["negative_ids"])) == 2
    assert set(result["positive_ids"]).isdisjoint(result["negative_ids"])
