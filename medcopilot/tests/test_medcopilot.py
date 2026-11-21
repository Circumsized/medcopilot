import pytest

from medcopilot.agents.triage import TriageAgent
from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.workflow.agent_graph import AgentGraph
from medcopilot.workflow.orchestrator import MedicalConsultationWorkflow
from medcopilot.workflow.safety_guard import MedicalSafetyGuard


def test_clinical_chunker():
    text = (
        "【药品名称】阿莫西林胶囊\n"
        "【适应症】用于敏感菌所致的呼吸道感染。\n"
        "【禁忌】青霉素过敏患者禁用。"
    )
    docs = ClinicalChunker.chunk_drug_label(text, drug_name="阿莫西林胶囊")
    assert len(docs) >= 2
    assert any(d.metadata.get("is_contraindication") is True for d in docs)
    assert any("呼吸道感染" in d.page_content for d in docs)


def test_hybrid_retriever():
    text = (
        "【药品名称】布洛芬\n"
        "【禁忌】消化道溃疡禁用。"
    )
    docs = ClinicalChunker.chunk_drug_label(text, drug_name="布洛芬")
    retriever = HybridRetriever(documents=docs)
    results = retriever.retrieve("消化道溃疡可以吃吗", top_n=1)
    assert len(results) == 1
    assert "布洛芬" in results[0].page_content


def test_safety_guard():
    emergency_q = "医生救命！我家里人突发心肌梗死而且吐血了！"
    is_emer, alert = MedicalSafetyGuard.check_emergency(emergency_q)
    assert is_emer is True
    assert "120" in alert

    normal_q = "头痛可以吃布洛芬吗？"
    is_emer, _ = MedicalSafetyGuard.check_emergency(normal_q)
    assert is_emer is False


@pytest.mark.parametrize(
    "query",
    ["突发心梗怎么办", "老人中风了嘴歪了", "孩子高热惊厥抽过去了", "想割腕", "误食安眠药中毒了"],
)
def test_safety_guard_covers_colloquial_terms(query):
    is_emer, alert = MedicalSafetyGuard.check_emergency(query)
    assert is_emer is True
    assert "120" in alert


def test_grounding_alignment_flags_unrelated_answer():
    ratio, grounded = MedicalSafetyGuard.check_groundedness(
        "今天天气很好，建议散步。",
        ["【禁忌】活动期消化道溃疡及出血患者禁用。"],
        threshold=0.15,
    )
    assert ratio == 0.0
    assert grounded is False


def test_agent_graph_low_confidence_query_triggers_warning():
    graph = AgentGraph(
        retriever=HybridRetriever(
            documents=ClinicalChunker.chunk_drug_label(
                "【药品名称】阿莫西林胶囊\n【禁忌】青霉素过敏患者禁用。",
                drug_name="阿莫西林胶囊",
            )
        )
    )
    result = graph.run("asdfgh 无关词")
    assert result["intent"] == TriageAgent.MEDICAL
    assert result["sources"] == []
    assert result["abstained"] is True
    assert "无法提供具体用药建议" in result["answer"]


def test_orchestration_workflow():
    workflow = MedicalConsultationWorkflow()
    # 触发急救防护
    res = workflow.run("心肌梗死怎么办？")
    assert res["is_emergency"] is True
    assert "120" in res["answer"]

    # 正常问诊（走模拟 LLM）：模拟回复不携带本地语料证据，必然走弃答路径。
    # 如实固定该行为（真实"摄取→检索命中"契约由 test_medcopilot_ui_api_client
    # 的可控 LLM 测试钉死），避免本测试在摄取失效时假绿。
    res_normal = workflow.run("轻微感冒流涕如何处理？")
    assert res_normal["is_emergency"] is False
    assert res_normal["abstained"] is True
    assert res_normal["degradations"] == []
    assert "免责合规提示" in res_normal["answer"]


def test_triage_english_word_boundaries():
    # 'diet' should not trigger 'die' emergency
    diet_q = "高血压患者日常 diet 怎么安排？"
    assert TriageAgent.classify(diet_q) == TriageAgent.MEDICAL

    # 'die' standalone should trigger emergency
    die_q = "I want to die immediately"
    assert TriageAgent.classify(die_q) == TriageAgent.EMERGENCY


def test_ocr_tool_fuzzy_and_line_matching():
    from medcopilot.agents.ocr_tool import OCRTool
    ocr = OCRTool()
    drug_dict = ["阿莫西林胶囊", "布洛芬缓释胶囊", "阿司匹林肠溶片"]

    # Line contains drug name with noise
    matched = ocr.match_drug_name(["生产批号: 20240101", "通用名: 阿莫西林胶囊 0.5g", "口服"], drug_dict)
    assert matched == "阿莫西林胶囊"

    # Fuzzy match with slight OCR error
    matched_fuzzy = ocr.match_drug_name(["布洛劳缓释胶囊"], drug_dict, cutoff=0.6)
    assert matched_fuzzy == "布洛芬缓释胶囊"


def test_vector_store_empty_handling():
    from medcopilot.rag.vector_store import ChromaDenseEncoder
    class MockCollection:
        def count(self):
            return 0
    encoder = ChromaDenseEncoder(collection=MockCollection(), embedding_fn=lambda texts: [[0.0]])
    results = encoder.query_similar("任意查询", top_n=5)
    assert results == []
