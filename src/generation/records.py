from __future__ import annotations


FORMAT_SKIP_POLICY = "skip_record_if_any_response_sections_invalid"


def has_valid_response_format(record: dict) -> bool:
    """Return whether every response needed by the UQ pipeline is structured."""
    greedy = record.get("greedy")
    samples = record.get("samples")
    return (
        isinstance(greedy, dict)
        and greedy.get("sections_valid") is True
        and isinstance(samples, list)
        and bool(samples)
        and all(
            isinstance(sample, dict) and sample.get("sections_valid") is True
            for sample in samples
        )
    )
