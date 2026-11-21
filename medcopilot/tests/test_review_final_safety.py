"""Fail-closed workflow and isolated logging regression tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from medcopilot.rag.clinical_chunker import Document
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.rag.reranker import Reranker
from medcopilot.workflow.agent_graph import AgentGraph, ConsultationState, Session


class RecordingLLM:
    def __init__(self, answer="测试药甲每次999毫克，一天9次。"):
        self.calls = 0
        self.answer = answer

    def invoke(self, prompt):
        self.calls += 1
        return self.answer


@pytest.mark.parametrize("query", ["头痛如何治疗？", "这个药怎么吃？"])
def test_no_evidence_never_calls_llm(query):
    llm = RecordingLLM()
    graph = AgentGraph(retriever=HybridRetriever([]), llm=llm, reranker=Reranker(enabled=False))
    session = Session()
    result = graph.run(query, drug_context="OCR: 测试药甲", session=session)
    assert llm.calls == 0
    assert result["abstained"] is True
    assert result["sources"] == []
    assert "999" not in result["answer"]
    assert "通用大模型知识" not in result["answer"]
    assert session.turns[0]["answer"] == result["answer"]


def test_reviewer_replaces_unsupported_treatment():
    llm = RecordingLLM()
    graph = AgentGraph(llm=llm, reranker=Reranker(enabled=False))
    state = ConsultationState(
        query="头痛如何治疗？", answer=llm.answer,
        evidence=[Document("头痛咨询资料：药盒避光保存。", {"chunk_id": "x"})],
    )
    checked = graph.reviewer(state)
    assert checked.abstained is True
    assert "999" not in checked.answer
    assert "测试药甲" not in checked.answer


def test_insufficient_evidence_abstains_before_generation():
    """W3.3: evidence_floor 已退役(RRF 归一化分数不是校准置信度,实测完全无关的
    查询也得 0.49~0.99,高分弱证据可绕过)。弃答判据由 SufficiencyGate 承担:
    query 锚点未被证据覆盖 ⇒ 不调 LLM 直接弃答——本测试固化该失效点的修复。"""
    class HighScoreIrrelevantRetriever:
        def retrieve_with_scores(self, query, top_n):
            return [(Document("头痛资料。"), 0.99)]

    llm = RecordingLLM()
    graph = AgentGraph(
        retriever=HighScoreIrrelevantRetriever(), llm=llm, reranker=Reranker(enabled=False),
        sufficiency_dry_run=False,  # 硬门路径;默认影子模式只记录不弃答
    )
    result = graph.run("膝关节置换术后康复训练方案？")
    assert result["abstained"] is True
    assert llm.calls == 0
    assert result["sources"] == []
    # 弃答理由可分类:sufficiency 门给出机器可读 reason
    assert any(d.startswith("sufficiency:") for d in result["degradations"])


def test_partially_grounded_answer_does_not_leak_unsupported_sentence():
    graph = AgentGraph(reranker=Reranker(enabled=False))
    state = ConsultationState(
        query="药品怎么保存？", answer="请避光保存。测试药甲每次999毫克，一天9次。",
        evidence=[Document("请避光保存。")],
    )
    assert graph.reviewer(state).abstained is True
    assert "999" not in state.answer


def test_supported_answer_and_sources_remain_available():
    text = "药品应避光保存。"
    graph = AgentGraph(
        retriever=HybridRetriever([Document(text, {"chunk_id": "approved-local"})]),
        llm=RecordingLLM(text), reranker=Reranker(enabled=False),
    )
    result = graph.run("药品怎么保存？")
    assert result["abstained"] is False
    assert text in result["answer"]
    assert result["sources"][0]["chunk_id"] == "approved-local"


@pytest.mark.parametrize("evidence,answer", [
    ("禁止每日服用9片。", "每日服用9片。"),
    ("不得自行停药。", "自行停药。"),
    ("成人每次0.3g。", "成人每次3g。"),
])
def test_truncated_negation_or_changed_dose_is_rejected(evidence, answer):
    graph = AgentGraph(reranker=Reranker(enabled=False))
    state = ConsultationState(query="请核对说明书", answer=answer, evidence=[Document(evidence)])
    assert graph.reviewer(state).abstained is True


@pytest.mark.parametrize("mode", ["disabled", "relocated", "unwritable"])
def test_logger_import_in_isolated_process(tmp_path, mode):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONIOENCODING="utf-8", LOG_LEVEL="INFO")
    logfile = tmp_path / "custom.log"
    env["LOG_FILE"] = "" if mode == "disabled" else str(tmp_path if mode == "unwritable" else logfile)
    result = subprocess.run(
        [sys.executable, "-c", "from medcopilot.common.logger import logger; logger.info('ISOLATED_LOG_SENTINEL')"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ISOLATED_LOG_SENTINEL" in result.stdout
    if mode == "relocated":
        assert "ISOLATED_LOG_SENTINEL" in logfile.read_text(encoding="utf-8")
    else:
        assert not logfile.exists()
        assert not (tmp_path / "logs").exists()
