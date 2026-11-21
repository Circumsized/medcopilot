"""Offline regressions for the explicitly lexical, source-scoped evaluation proxy."""

import json
import socket
from types import SimpleNamespace

import pytest

from medcopilot.agents.triage import TriageAgent
from medcopilot.common.config import config
from medcopilot.llm.unified_llm import UnifiedLLM
from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.reranker import Reranker
from medcopilot.scripts import eval_rag
from medcopilot.workflow.agent_graph import AgentGraph

SUPPORTED = "活动期消化道溃疡及出血患者禁用。"
UNSUPPORTED = "孕妇可以放心服用此药且无需调整剂量。"
CASE = {
    "corpus": "【适应症】用于缓解轻至中度疼痛。\n【禁忌】" + SUPPORTED,
    "drug": "测试药品",
    "question": "活动期消化道溃疡及出血患者禁用吗？",
}


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("Evaluation regressions must not access the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(config, "rerank_enabled", False)


def evaluate(answer, *, source_indices=(1,), case=None, sources=None, **status):
    case = CASE if case is None else case
    docs = ClinicalChunker.chunk_drug_label(case["corpus"], drug_name=case["drug"])
    result = {
        "answer": answer,
        "sources": [dict(docs[index].metadata) for index in source_indices] if sources is None else sources,
        "intent": TriageAgent.MEDICAL,
        "is_emergency": False,
        "abstained": False,
        **status,
    }
    # Only replace the graph boundary: exercise the real chunking and evaluator.
    graph = SimpleNamespace(run=lambda question: result)
    return eval_rag.evaluate_case(graph, case)


def test_no_retrieved_sources_cannot_borrow_corpus_support():
    metrics = evaluate(SUPPORTED, sources=[])
    assert metrics["faithfulness"] == 0.0
    assert metrics["context_relevance"] == 0.0


def test_unsupported_contraindication_is_not_supported_by_keyword():
    metrics = evaluate("所有儿童都禁用此药。")
    assert metrics["faithfulness"] == 0.0


def test_disclaimer_alone_is_not_a_faithful_answer():
    metrics = evaluate("医疗免责声明：用药请咨询医生。")
    assert metrics["faithfulness"] == 0.0


def test_verbatim_sentence_from_used_evidence_is_supported():
    metrics = evaluate(SUPPORTED)
    assert metrics["faithfulness"] == 1.0
    assert metrics["abstained"] is False
    assert "extractive" in metrics["metric_protocol"]
    assert "proxy" in metrics["metric_protocol"]
    assert metrics["chunks"] == 2


def test_only_unused_document_supports_answer():
    assert evaluate(SUPPORTED, source_indices=(0,))["faithfulness"] == 0.0


def test_context_relevance_uses_only_used_document():
    case = {"corpus": "甲乙丙丁用于缓解头痛。", "drug": "测试药品", "question": "青霉素过敏"}
    case["corpus"] = "【适应症】" + case["corpus"] + "\n【禁忌】青霉素过敏患者禁用。"
    metrics = evaluate("甲乙丙丁用于缓解头痛。", case=case, source_indices=(0,))
    # The injected document headers also contain none of these query characters.
    assert metrics["context_relevance"] == 0.0


@pytest.mark.parametrize("sources", [
    [{"chunk_id": "unknown-id", "page_content": SUPPORTED}],
    [{"page_content": SUPPORTED}],
    [{"chunk_id": None}],
    [{"chunk_id": ["malformed"]}],
])
def test_unknown_or_invalid_source_ids_cannot_supply_evidence(sources):
    metrics = evaluate(SUPPORTED, sources=sources)
    assert metrics["faithfulness"] == 0.0
    assert metrics["context_relevance"] == 0.0


def test_source_payload_cannot_override_known_chunk_content():
    docs = ClinicalChunker.chunk_drug_label(CASE["corpus"], drug_name=CASE["drug"])
    sources = [{"chunk_id": docs[0].metadata["chunk_id"], "page_content": SUPPORTED}]
    assert evaluate(SUPPORTED, sources=sources)["faithfulness"] == 0.0


def test_abstained_flag_overrides_even_verbatim_support():
    metrics = evaluate(SUPPORTED, abstained=True)
    assert metrics["faithfulness"] == 0.0
    assert metrics["abstained"] is True


@pytest.mark.parametrize("status", [
    {"intent": TriageAgent.NON_MEDICAL},
    {"intent": TriageAgent.EMERGENCY, "is_emergency": True},
    {"refused": True},
])
def test_refusal_and_emergency_are_not_faithful_medical_answers(status):
    assert evaluate(SUPPORTED, **status)["faithfulness"] == 0.0


@pytest.mark.parametrize("prefix", [
    "【模拟诊疗意见】", "【系统错误提示】", "【系统提示：流式传输已中断】", "无法提供具体用药建议。",
])
def test_nonanswer_markers_zero_entire_response_even_with_copied_evidence(prefix):
    assert evaluate(prefix + "\n" + SUPPORTED)["faithfulness"] == 0.0


@pytest.mark.parametrize("answer", [
    "免责声明：本回答仅供参考，请咨询医生。",
    "本回答不能替代专业诊疗。",
    "请务必遵循正规医疗机构专业医师指导。",
])
def test_even_copied_disclaimers_cannot_earn_support(answer):
    case = {"corpus": answer, "drug": "测试药品", "question": "头痛用药"}
    assert evaluate(answer, case=case, source_indices=(0,))["faithfulness"] == 0.0


def test_unsupported_sentence_remains_in_denominator():
    assert evaluate(SUPPORTED + UNSUPPORTED)["faithfulness"] == 0.5


def test_disclaimer_does_not_add_support_to_extractive_answer():
    assert evaluate(SUPPORTED + "\n本回答仅供参考，请咨询医生。")["faithfulness"] == 0.5


@pytest.mark.parametrize("answer", ["禁。", "活动期消化道溃疡及出血患者可以用药。", "出血患者禁用。"])
def test_character_overlap_and_partial_sentences_are_not_extractive_support(answer):
    assert evaluate(answer)["faithfulness"] == 0.0


def test_normalization_preserves_decimal_dose_and_fullwidth_digits():
    case = {"corpus": "成人每次服用０．３ｇ。", "drug": "测试药品", "question": "成人用药"}
    assert evaluate("成人每次服用0.3g。", case=case, source_indices=(0,))["faithfulness"] == 1.0
    assert evaluate("成人每次服用3g。", case=case, source_indices=(0,))["faithfulness"] == 0.0


def test_empty_answer_has_no_support():
    assert evaluate("")["faithfulness"] == 0.0


def test_empty_suite_does_not_pass(monkeypatch, capsys):
    monkeypatch.setattr(eval_rag, "DEFAULT_SUITE", [])
    monkeypatch.setattr(eval_rag.sys, "argv", ["eval_rag.py"])
    with pytest.raises(SystemExit) as exc:
        eval_rag.main()
    assert exc.value.code != 0
    assert "PASS" not in capsys.readouterr().out


def test_cli_preserves_custom_thresholds(monkeypatch, capsys):
    monkeypatch.setattr(eval_rag, "DEFAULT_SUITE", [CASE])
    monkeypatch.setattr(eval_rag, "evaluate_case", lambda *args: {"faithfulness": 0.8, "context_relevance": 0.6})
    monkeypatch.setattr(eval_rag.sys, "argv", [
        "eval_rag.py", "--threshold-faithfulness", "0.9", "--threshold-context", "0.55",
    ])
    with pytest.raises(SystemExit) as exc:
        eval_rag.main()
    assert exc.value.code == 1
    assert "faithfulness>=0.9, context>=0.55" in capsys.readouterr().out


def test_default_mock_suite_honestly_fails(monkeypatch, capsys):
    graph = AgentGraph(llm=UnifiedLLM(provider="mock"), reranker=Reranker(enabled=False))
    monkeypatch.setattr(eval_rag, "AgentGraph", lambda: graph)
    monkeypatch.setattr(eval_rag.sys, "argv", ["eval_rag.py"])
    with pytest.raises(SystemExit) as exc:
        eval_rag.main()
    assert exc.value.code == 1
    output = capsys.readouterr().out
    report_text, status = output.rsplit("\n", 2)[:2]
    report = json.loads(report_text)
    assert status.startswith("FAIL")
    assert all(row["faithfulness"] == 0.0 for row in report)
    assert all(isinstance(row["abstained"], bool) for row in report)
    assert all("proxy" in row["metric_protocol"] for row in report)
