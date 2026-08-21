from src.models.runtime import (
    replay_batch_size,
    vllm_max_model_len,
    vllm_max_num_seqs,
)


def test_32_gib_production_sizes() -> None:
    assert replay_batch_size(32.0) == 5
    assert vllm_max_num_seqs(32.0) == 8


def test_smaller_gpus_scale_down() -> None:
    assert replay_batch_size(23.0) == 2
    assert vllm_max_num_seqs(23.0) == 4


def test_qwen_uses_audited_long_multimodal_context() -> None:
    assert vllm_max_model_len("qwen2_5_vl") == 18_000
    assert vllm_max_model_len("llava_1_5") == 4096
    assert vllm_max_model_len("internvl3_5_original") == 4096
