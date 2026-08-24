from scripts.extract_quartile_luh_subsets import select_quartile_luh


def _row(sample_id: str, score: float, hallucination: bool, dataset: str = "vilp"):
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "group_id": sample_id,
        "hallucination": hallucination,
        "scores": {
            "perplexity": score,
            "semantic_entropy": score,
            "umpire": score,
        },
    }


def test_selects_hallucinations_at_or_below_h0_quartile():
    rows = [
        _row("h0-1", 1.0, False),
        _row("h0-2", 2.0, False),
        _row("h0-3", 3.0, False),
        _row("h0-4", 4.0, False),
        _row("h1-low", 1.5, True),
        _row("h1-tie", 1.75, True, "mmvet"),
        _row("h1-high", 2.0, True),
    ]

    result = select_quartile_luh(rows, "perplexity", 0.25)

    assert result["threshold"] == 1.75
    assert result["sample_ids"] == ["h1-low", "h1-tie"]
    assert result["n_h0"] == 4
    assert result["n_h1"] == 3
    assert result["n_selected"] == 2
    assert result["dataset_counts"] == {
        "vilp": 1,
        "hallusionbench": 0,
        "mmvet": 1,
    }


def test_subset_size_is_not_forced_to_alpha_fraction():
    rows = [
        *[_row(f"h0-{index}", float(index), False) for index in range(4)],
        *[_row(f"h1-{index}", 0.0, True) for index in range(7)],
    ]

    result = select_quartile_luh(rows, "semantic_entropy", 0.25)

    assert result["n_selected"] == 7
    assert result["luh_share"] == 1.0
