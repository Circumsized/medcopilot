"""W3.1 安全章节召回契约(section-aware retrieval + reviewer 在场门)。

对应上一轮专家评估的临床危害最大单项:grounding gate 只校验"答案⊆证据",
不校验"证据⊇禁忌"——禁忌块被 top-k 挤出后,LLM 照抄适应症即可通过两道校验。
修复 = 触发语境识别 + 保护位强制并入 + reviewer 在场检查(fail-closed)。
"""

from __future__ import annotations

from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.rag.reranker import Reranker
from medcopilot.safety.section_policy import ContraindicationTrigger
from medcopilot.workflow.agent_graph import AgentGraph

ULCER_QUERY = "我有胃溃疡，最近头痛想买点止痛药，哪个好？"
ENCYCLOPEDIA_QUERY = "布洛芬缓释胶囊能缓解哪些疼痛？"


class LabelledLLM:
    """可控 LLM:返回注入的回答文本,记录调用。"""

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = 0

    def invoke(self, prompt, **kwargs):
        self.calls += 1
        return self.reply

    def _should_mock(self):
        return True

    def _mock_reply(self, prompt, **kwargs):
        return self.invoke(prompt)


def _graph(label: str, drug: str, llm, **overrides) -> AgentGraph:
    kwargs = dict(
        retriever=HybridRetriever(documents=ClinicalChunker.chunk_drug_label(label, drug_name=drug)),
        llm=llm, reranker=Reranker(enabled=False),
    )
    kwargs.update(overrides)
    return AgentGraph(**kwargs)


def test_trigger_detects_comorbidity_and_decision_intent():
    ctx = ContraindicationTrigger.assess(ULCER_QUERY)
    assert ctx.triggered is True
    assert "胃溃疡" in ctx.conditions and "止痛药" in ctx.conflicts
    assert ctx.decision_intent is True
    assert ContraindicationTrigger.assess("布洛芬缓释胶囊的规格是什么").triggered is False


def test_contraindication_chunk_forced_into_evidence():
    label = (
        "【药品名称】布洛芬缓释胶囊\n"
        "【适应症】用于缓解轻至中度疼痛如头痛、关节痛、痛经。\n"
        "【用法用量】口服，成人一次1粒，一日2次。\n"
        "【禁忌】活动期消化道溃疡患者禁用。"
    )
    # 制造最坏情形:BM25 语义让适应症/用法用量排前、禁忌排后(top-k 挤出)。
    graph = _graph(label, "布洛芬缓释胶囊", LabelledLLM("请遵医嘱。"))
    result = graph.run(ULCER_QUERY)
    sections = {s["section"] for s in result["sources"]}
    assert "禁忌" in sections, "保护位必须把禁忌块送进最终证据"


def test_decision_intent_without_safety_section_abstains():
    label = (
        "【药品名称】某药片\n"
        "【适应症】用于缓解轻至中度疼痛。\n"
        "【用法用量】口服，成人一次1片，一日2次。"
    )
    # 回复为证据逐字句(extractive 切分后与证据句逐字节相等):
    # grounding/extractive 均通过,唯一拦截点落在安全章节门
    llm = LabelledLLM("用于缓解轻至中度疼痛。")
    graph = _graph(label, "某药片", llm)
    result = graph.run("胃不好的人能吃某药片止痛吗？")
    assert result["abstained"] is True, "触发语境 + 证据无安全章节 ⇒ 必须弃答(MedRGB sufficiency)"
    assert any(d.startswith("safety-section:") for d in result["degradations"])


def test_encyclopedic_query_not_blocked():
    """百科式提问不触发安全门:证据含适应症即可正常作答路径,不因门误伤。"""
    label = (
        "【药品名称】布洛芬缓释胶囊\n"
        "【适应症】用于缓解轻至中度疼痛如头痛、关节痛、痛经。"
    )
    llm = LabelledLLM("【适应症】用于缓解轻至中度疼痛如头痛、关节痛、痛经。")
    graph = _graph(label, "布洛芬缓释胶囊", llm)
    result = graph.run(ENCYCLOPEDIA_QUERY)
    # 百科查询未触发安全语境(无共病/冲突/决策意图词)⇒ 不因 safety-section 弃答
    assert not any(d.startswith("safety-section:") for d in result["degradations"])


def test_protected_slot_dedup_and_cap():
    label = (
        "【药品名称】布洛芬缓释胶囊\n"
        "【适应症】用于缓解轻至中度疼痛。\n"
        "【禁忌】活动期消化道溃疡患者禁用。\n"
        "【药物相互作用】避免与其他抗凝药同用。\n"
        "【注意事项】服药期间不得饮酒。"
    )
    graph = _graph(label, "布洛芬缓释胶囊", LabelledLLM("请遵医嘱。"))
    result = graph.run(ULCER_QUERY)
    assert len(result["sources"]) <= 5, "证据总数不超过 evidence_top_n"
    chunk_ids = [s["chunk_id"] for s in result["sources"]]
    assert len(chunk_ids) == len(set(chunk_ids)), "保护位不得重复追加同一 chunk"


def test_ocr_drug_context_triggers_safety_recall():
    """OCR 命中药品(drug_context) + 触发词 ⇒ 同样强制安全章节召回。"""
    label = (
        "【药品名称】布洛芬缓释胶囊\n"
        "【适应症】用于缓解轻至中度疼痛。\n"
        "【禁忌】活动期消化道溃疡患者禁用。"
    )
    graph = _graph(label, "布洛芬缓释胶囊", LabelledLLM("请遵医嘱。"))
    result = graph.run("这个药我能长期吃吗", drug_context="识别药品: 布洛芬缓释胶囊（置信匹配）")
    sections = {s["section"] for s in result["sources"]}
    assert "禁忌" in sections


def test_safety_gate_disabled_restores_legacy_behavior():
    """开关关闭(SAFETY_SECTION_GATE_ENABLED=0)时,无安全章节不弃答——
    为影子评测/灰度对照保留逃生口,配置项走 AppConfig。"""
    label = (
        "【药品名称】某药片\n"
        "【适应症】用于缓解轻至中度疼痛。\n"
        "【用法用量】口服，成人一次1片，一日2次。"
    )
    graph = _graph(label, "某药片", LabelledLLM("用于缓解轻至中度疼痛。"), safety_section_gate_enabled=False)
    result = graph.run("胃不好的人能吃某药片止痛吗？")
    assert not any(d.startswith("safety-section:") for d in result["degradations"])
