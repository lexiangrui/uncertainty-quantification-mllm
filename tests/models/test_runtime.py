from src.models.runtime import replay_batch_size, vllm_max_num_seqs


def test_32_gib_production_sizes() -> None:
    assert replay_batch_size(32.0) == 5
    assert vllm_max_num_seqs(32.0) == 8


def test_smaller_gpus_scale_down() -> None:
    assert replay_batch_size(23.0) == 2
    assert vllm_max_num_seqs(23.0) == 4
