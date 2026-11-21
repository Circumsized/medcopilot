from __future__ import annotations

import pytest

from medcopilot.evaluation.rag_official_adapter import (
    PROTOCOL_VERSION,
    EvaluationSample,
    OfficialRAGAdapter,
    evaluate_consultations,
    sample_from_consultation,
)
from medcopilot.workflow.agent_graph import ConsultationResult


@pytest.fixture
def result():
    return {
        "answer": "依据说明书回答。",
        "sources": [
            {"chunk_id": "local-1", "section": "禁忌", "text": "不要信任此字段"},
            {"chunk_id": "unknown", "section": "适应症"},
            {"chunk_id": "local-1", "section": "禁忌"},
        ],
        "intent": "medical",
        "abstained": False,
        "is_emergency": False,
        "policy_version": 7,
    }


def test_consultation_maps_only_source_scoped_local_context(result):
    sample = sample_from_consultation(
        "这个药有什么禁忌？",
        result,
        source_contexts={"local-1": "本地说明书禁忌内容", "unknown": "另一个本地块"},
    )
    assert isinstance(sample, EvaluationSample)
    assert sample.retrieved_contexts == ("本地说明书禁忌内容", "另一个本地块")
    assert sample.metadata["source_count"] == 2
    assert sample.metadata["source_sections"] == ("禁忌", "适应症", "禁忌")
    assert "text" not in sample.retrieved_contexts


def test_unknown_or_malformed_sources_are_ignored(result):
    result["sources"] = [None, "bad", {"section": "禁忌"}, {"chunk_id": "missing"}]
    sample = sample_from_consultation("q", result, source_contexts={"local": "text"})
    assert sample.retrieved_contexts == ()


def test_invalid_inputs_are_rejected(result):
    with pytest.raises(ValueError):
        sample_from_consultation("", result, source_contexts={})
    with pytest.raises(TypeError):
        sample_from_consultation("q", object(), source_contexts={})


def test_adapter_has_stable_not_configured_protocol(monkeypatch):
    monkeypatch.setattr(
        "medcopilot.evaluation.rag_official_adapter.find_spec",
        lambda _: None,
    )
    adapter = OfficialRAGAdapter("ragas")
    report = adapter.evaluate([EvaluationSample("q", "a", ("c",))])
    assert report == {
        "protocol_version": PROTOCOL_VERSION,
        "adapter": "ragas",
        "offline_only": True,
        "status": "not-configured",
        "sample_count": 1,
        "scores": {},
        "error": "optional dependency not installed: ragas",
    }


def test_runner_scores_are_normalized_and_offline():
    adapter = OfficialRAGAdapter(
        "deepeval",
        runner=lambda samples: {"faithfulness": 0.8, "context_relevance": 0.5},
    )
    report = adapter.evaluate([EvaluationSample("q", "a", ("c",))])
    assert report["status"] == "ok"
    assert report["offline_only"] is True
    assert report["scores"] == {"faithfulness": 0.8, "context_relevance": 0.5}


def test_consultation_result_instance_is_supported():
    result = ConsultationResult(
        answer="answer", sources=[{"chunk_id": "c1", "section": "适应症"}],
        is_emergency=False, intent="medical", abstained=False,
        degradations=[], policy_version=1, dosage_gate={},
    )
    sample = sample_from_consultation("q", result, source_contexts={"c1": "local"})
    assert sample.retrieved_contexts == ("local",)


def test_non_finite_or_out_of_range_scores_are_not_accepted():
    for value in (float("nan"), 1.1):
        report = OfficialRAGAdapter("ragas", runner=lambda _, value=value: {"faithfulness": value}).evaluate(
            [EvaluationSample("q", "a", ("c",))]
        )
        assert report["status"] == "not-configured"
        assert "scores" in report["error"] or "scores" in str(report["error"])


    adapter = OfficialRAGAdapter("ragas", runner=lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
    report = adapter.evaluate([EvaluationSample("q", "a", ("c",))])
    assert report["status"] == "not-configured"
    assert report["error"] == "RuntimeError: boom"


def test_remote_mode_and_unknown_provider_rejected():
    with pytest.raises(ValueError, match="offline_only"):
        OfficialRAGAdapter("ragas", offline_only=False)
    with pytest.raises(ValueError, match="provider"):
        OfficialRAGAdapter("langchain")


def test_convenience_api_maps_consultation_without_graph_changes(monkeypatch, result):
    monkeypatch.setattr(
        "medcopilot.evaluation.rag_official_adapter.find_spec",
        lambda _: None,
    )
    report = evaluate_consultations(
        "ragas",
        [("q", result, {"local-1": "label"})],
    )
    assert report["sample_count"] == 1
    assert report["offline_only"] is True
