"""W4 embedding identity/Chinese model contract tests (offline doubles only)."""
from __future__ import annotations

import pytest

from medcopilot.rag.vector_store import MedBgeM3EmbeddingFunction


def test_embedding_function_config_round_trip_without_loading_model():
    fn = MedBgeM3EmbeddingFunction(model_name="BAAI/bge-m3", batch_size=8, normalize=True)
    rebuilt = MedBgeM3EmbeddingFunction.build_from_config(fn.get_config())
    assert rebuilt.get_config() == fn.get_config()
    assert rebuilt.model_name == "BAAI/bge-m3"


def test_embedding_function_is_lazy():
    fn = MedBgeM3EmbeddingFunction(model_name="missing/offline-model")
    assert fn._model is None
    assert fn.failed is False


def test_embedding_function_failure_is_explicit_and_retry_free(monkeypatch):
    fn = MedBgeM3EmbeddingFunction(model_name="missing/offline-model")
    monkeypatch.setattr(fn, "_load", lambda: None)
    with pytest.raises(RuntimeError, match="embedding model unavailable"):
        fn(["中文药品禁忌"])


def test_embedding_function_supports_string_and_batch_and_caches_dimension(monkeypatch):
    class Model:
        def encode(self, texts, **kwargs):
            return [[float(len(texts)), 1.0, 2.0] for _ in texts]

    fn = MedBgeM3EmbeddingFunction(model_name="offline", batch_size=2)
    monkeypatch.setattr(fn, "_load", lambda: Model())
    assert fn("药品") == [1.0, 1.0, 2.0]
    assert fn(["药品", "禁忌"]) == [[2.0, 1.0, 2.0], [2.0, 1.0, 2.0]]
    assert fn.embedding_dim == 3
