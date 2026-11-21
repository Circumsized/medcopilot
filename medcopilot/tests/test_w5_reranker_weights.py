"""W5.3: Reranker 权重状态面与双检锁加载（全离线，绝不下载权重）。"""

from __future__ import annotations

import threading

import pytest

from medcopilot.common.config import config
from medcopilot.rag.clinical_chunker import Document
from medcopilot.rag.reranker import RERANK_WEIGHTS_UNAVAILABLE, Reranker

# The re-ranker weight paths are exercised only where the optional heavy stack
# (sentence-transformers/torch) is installed; CI installs the rag lock on the
# 3.12 job only, so older interpreters skip the whole module.
pytest.importorskip("sentence_transformers")

DOCS = [
    Document(page_content="【禁忌】活动期消化道溃疡及出血患者禁用。", metadata={"chunk_id": "a"}),
    Document(page_content="【适应症】用于缓解轻至中度疼痛。", metadata={"chunk_id": "b"}),
]


def _patch_cross_encoder(monkeypatch, *, constructed=0, lock=None):
    """把 sentence_transformers.CrossEncoder 换成记录构造参数的计数替身。"""
    import sentence_transformers

    guard = lock or threading.Lock()
    state = {"constructed": 0, "calls": []}

    class _FakeCrossEncoder:
        def __init__(self, model_name, **kwargs):
            with guard:
                state["constructed"] += 1
            state["calls"].append({"model_name": model_name, **kwargs})

        def predict(self, pairs):
            return [float(len(p[0])) for p in pairs]

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _FakeCrossEncoder)
    return state


# ------------------------------------------------------------------ 状态面


def test_disabled_reranker_reports_disabled_state():
    reranker = Reranker(enabled=False)
    assert reranker.weights_state == "disabled"
    assert reranker.mode == "disabled"
    status = reranker.weights_status()
    assert status["state"] == "disabled"
    assert status["degradation_code"] is None


def test_weights_state_is_unloaded_before_the_first_call():
    reranker = Reranker(enabled=True)
    assert reranker.weights_state == "unloaded"
    assert reranker.weights_status()["degradation_code"] is None


def test_successful_load_reports_local_weights(monkeypatch):
    state = _patch_cross_encoder(monkeypatch)
    reranker = Reranker(model_name="BAAI/bge-reranker-base")
    ranked = reranker.rerank("消化道溃疡", DOCS, top_n=2)

    assert state["constructed"] == 1
    assert reranker.weights_state == "local"
    assert reranker.mode == "cross-encoder"
    assert reranker.weights_status()["state"] == "local"
    assert ranked and ranked[0][0].metadata["chunk_id"] == "a"


def test_missing_weights_report_missing_with_an_explicit_degradation(monkeypatch):
    monkeypatch.setattr(
        "sentence_transformers.CrossEncoder",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no local snapshot")),
    )
    reranker = Reranker(model_name="BAAI/bge-reranker-base")
    ranked = reranker.rerank("消化道溃疡", DOCS, top_n=2)

    assert reranker.weights_state == "missing"
    assert reranker.mode == "lexical"
    assert reranker.weights_status()["degradation_code"] == RERANK_WEIGHTS_UNAVAILABLE
    assert "no local snapshot" in reranker.weights_status()["detail"]
    # 显式降级而不是崩溃：词法重排仍然给出顺序。
    assert ranked and ranked[0][0].metadata["chunk_id"] == "a"


def test_failure_is_sticky_and_does_not_retry_forever(monkeypatch):
    import sentence_transformers

    calls = {"n": 0}

    class _AlwaysFail:
        def __init__(self, *a, **k):
            calls["n"] += 1
            raise OSError("offline")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _AlwaysFail)
    reranker = Reranker(model_name="BAAI/bge-reranker-base")
    for _ in range(3):
        reranker.rerank("q", DOCS, top_n=1)
    assert calls["n"] == 1, "加载失败后不得反复重试"


# ------------------------------------------------------------------ 双检锁


def test_load_is_double_checked_under_concurrency(monkeypatch):
    lock = threading.Lock()
    state = _patch_cross_encoder(monkeypatch, lock=lock)

    class SlowCrossEncoder:
        def __init__(self, model_name, **kwargs):
            with lock:
                state["constructed"] += 1
            state["calls"].append({"model_name": model_name, **kwargs})

        def predict(self, pairs):
            return [0.5 for _ in pairs]

    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", SlowCrossEncoder)
    reranker = Reranker(model_name="BAAI/bge-reranker-base")
    errors: list[Exception] = []

    def _run():
        try:
            reranker.rerank("消化道溃疡", DOCS, top_n=2)
        except Exception as exc:  # pragma: no cover - 仅在实现退步时触发
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert state["constructed"] == 1, "并发下模型必须只构造一次（双检锁）"
    assert reranker.weights_state == "local"


# ------------------------------------------------------------------ 离线优先


def test_loader_is_offline_first_with_the_configured_cache(monkeypatch):
    monkeypatch.setattr(config, "model_weights_offline", True)
    monkeypatch.setattr(config, "rerank_cache_dir", "/opt/hf-cache")
    state = _patch_cross_encoder(monkeypatch)

    reranker = Reranker(model_name="BAAI/bge-reranker-base")
    assert reranker.local_files_only is True
    assert reranker.cache_dir == "/opt/hf-cache"
    reranker.rerank("q", DOCS, top_n=1)

    call = state["calls"][0]
    assert call["model_name"] == "BAAI/bge-reranker-base"
    assert call["local_files_only"] is True
    assert call["cache_dir"] == "/opt/hf-cache"
    assert call["max_length"] == config.rerank_max_length


def test_offline_mode_can_be_relaxed_explicitly(monkeypatch):
    monkeypatch.setattr(config, "model_weights_offline", False)
    state = _patch_cross_encoder(monkeypatch)
    reranker = Reranker(model_name="BAAI/bge-reranker-base")
    reranker.rerank("q", DOCS, top_n=1)
    assert state["calls"][0]["local_files_only"] is False


def test_defaults_come_from_app_config():
    reranker = Reranker()
    assert reranker.model_name == config.reranker_model
    assert reranker.enabled is True
    assert reranker.local_files_only is config.model_weights_offline
