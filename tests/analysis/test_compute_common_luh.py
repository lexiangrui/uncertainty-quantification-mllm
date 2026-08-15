from scripts.analysis.compute_common_luh import average_rank_percentiles


def test_average_rank_percentiles_preserve_ties() -> None:
    result = average_rank_percentiles({"a": 0.0, "b": 1.0, "c": 1.0, "d": 3.0})

    assert result == {"a": 0.25, "b": 0.625, "c": 0.625, "d": 1.0}
