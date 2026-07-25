from types import SimpleNamespace

from src.models.huggingface import HuggingFaceMultimodalBackend


class PeftLikeModel:
    def __init__(self, base_model):
        self.base_model = base_model

    def get_base_model(self):
        return self.base_model


def backend_with(model):
    backend = object.__new__(HuggingFaceMultimodalBackend)
    backend.model = model
    return backend


def test_semantic_hook_uses_decoder_below_causal_lm_head() -> None:
    decoder = object()
    causal_lm = SimpleNamespace(model=decoder)
    backend = backend_with(PeftLikeModel(SimpleNamespace(language_model=causal_lm)))

    assert backend._semantic_embedding_module() is decoder


def test_semantic_hook_accepts_language_decoder_without_separate_head() -> None:
    decoder = object()
    backend = backend_with(PeftLikeModel(SimpleNamespace(language_model=decoder)))

    assert backend._semantic_embedding_module() is decoder
