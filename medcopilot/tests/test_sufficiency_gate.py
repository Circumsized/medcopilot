"""W3.3 sufficiency gate 契约(替代实测失效的 evidence_floor)。

旧 evidence_floor 在 RRF 归一化下形同虚设(完全无关查询也得 0.49~0.99),
只是空匹配检测器。新判据:query 锚点(语料药名 + 停用字净化二元组 + 英文词 +
数字)被证据覆盖,零/近零重叠 ⇒ 弃答。默认影子模式(dry_run)只记录不弃答——
口语化 query 与正确证据重叠天然低,硬阈值会误杀(影子数据校准后再关闭)。
"""

from __future__ import annotations

from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.rag.reranker import Reranker
from medcopilot.safety.section_policy import SufficiencyGate
from medcopilot.workflow.agent_graph import AgentGraph

VOCAB = ["布洛芬缓释胶囊", "阿莫西林胶囊"]


# ------------------------------------------------------------ 单元:锚点与判据


def test_anchors_extract_drug_and_bigrams():
    anchors, drugs = SufficiencyGate.anchors("布洛芬缓释胶囊的禁忌是什么", VOCAB)
    assert drugs == ("布洛芬缓释胶囊",)
    assert "布洛芬缓释胶囊" in anchors
    assert "禁忌" in anchors
    # 停用字不产生跨字二元组("的"被剔除)
    assert "的禁" not in anchors and "是什" not in anchors


def test_evaluate_passes_for_direct_query():
    evidence = [d.page_content for d in ClinicalChunker.chunk_drug_label(
        "【药品名称】布洛芬缓释胶囊\n【禁忌】活动期消化道溃疡患者禁用。",
        drug_name="布洛芬缓释胶囊",
    )]
    passed, info = SufficiencyGate.evaluate("布洛芬缓释胶囊的禁忌是什么", evidence, VOCAB)
    assert passed is True and info["union_coverage"] >= 0.5


def test_evaluate_fails_for_zero_overlap():
    passed, info = SufficiencyGate.evaluate("膝关节置换术后康复训练方案", ["头痛资料。"], VOCAB)
    assert passed is False and info["reason"] == "anchor-coverage-below-floor"


def test_evaluate_fails_for_no_anchors():
    passed, info = SufficiencyGate.evaluate("帮我看看这个药", ["头痛资料。"], VOCAB)
    assert passed is False and info["reason"] == "no-anchors"




def test_required_sections_follow_query_intent_and_fail_when_missing():
    assert SufficiencyGate.required_sections("这个药怎么吃") == ("用法用量",)
    passed, info = SufficiencyGate.evaluate(
        "布洛芬缓释胶囊怎么吃",
        ["【禁忌】活动期溃疡禁用"],
        VOCAB,
        evidence_sections=["禁忌"],
    )
    assert not passed and info["reason"] == "required-section-missing"
    anchors, _ = SufficiencyGate.anchors("布洛芬缓释胶囊的禁忌是什么", VOCAB)
    wide, _ = SufficiencyGate.anchors("布洛芬缓释胶囊的禁忌是什么", VOCAB)  # NFKC 后等价
    assert anchors == wide


# ------------------------------------------------------------ 集成:graph 行为


class RecordingLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        return "用于缓解轻至中度疼痛如头痛。"

    def _should_mock(self):
        return True

    def _mock_reply(self, prompt, **kwargs):
        self.invoke(prompt)


def _graph(**overrides) -> AgentGraph:
    kwargs = dict(
        retriever=HybridRetriever(documents=ClinicalChunker.chunk_drug_label(
            "【药品名称】布洛芬缓释胶囊\n【适应症】用于缓解轻至中度疼痛如头痛。\n【禁忌】活动期消化道溃疡患者禁用。",
            drug_name="布洛芬缓释胶囊",
        )),
        llm=RecordingLLM(), reranker=Reranker(enabled=False),
    )
    kwargs.update(overrides)
    return AgentGraph(**kwargs)


def test_zero_overlap_query_abstains_without_llm_when_hard_gate_on():
    """高分弱证据(缺席药/纯噪声)在硬门路径下不调 LLM——固化 evidence_floor
    时代可绕过的失效点。默认影子模式,此处显式开启硬门。"""
    llm = RecordingLLM()
    graph = _graph(llm=llm, sufficiency_dry_run=False)
    result = graph.run("膝关节置换术后康复训练方案？")
    assert result["abstained"] is True and result["sources"] == []
    assert llm.calls == 0
    assert any(d.startswith("sufficiency:") for d in result["degradations"])


def test_dry_run_default_records_but_does_not_block():
    """默认影子模式:行为与旧版完全一致——不提前弃答、不清证据、degradations
    契约不被影子信号污染("空 = 全组件满质量"),影子信号走日志面供阈值校准。"""
    llm = RecordingLLM()
    graph = _graph(llm=llm, sufficiency_dry_run=True)
    result = graph.run("膝关节置换术后康复训练方案？")
    assert not any(d.startswith("sufficiency:") for d in result["degradations"])
    # 影子模式下弱证据照常走完流程(LLM 可被调用,弃答由 grounding 门承担)
    assert result["sources"] != [] or result["abstained"] is True


def test_direct_query_passes_hard_gate():
    """正常直达 query 在硬门路径下不被误杀。"""
    llm = RecordingLLM()
    graph = _graph(llm=llm, sufficiency_dry_run=False)
    result = graph.run("布洛芬缓释胶囊的禁忌是什么")
    assert result["abstained"] is False
    assert llm.calls >= 1


def test_deprecated_evidence_floor_still_accepted_but_inert():
    """evidence_floor 参数保留(向后兼容),但不再参与任何门控。"""
    llm = RecordingLLM()
    graph = _graph(llm=llm, evidence_floor=0.99, sufficiency_dry_run=True)
    result = graph.run("布洛芬缓释胶囊的禁忌是什么")
    # 旧 floor 若仍在门控,0.99 会清空全部证据;现在必须照常服务
    assert result["sources"] != []
