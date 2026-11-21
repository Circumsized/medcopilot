from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.rag.reranker import Reranker
from medcopilot.workflow.agent_graph import AgentGraph

# ------------------------------- agent graph --------------------------------

def test_chunker_content_alignment_regression():
    """Guards the two-capture-group split bug: chunk content must carry real
    label text, not bare section names."""
    docs = ClinicalChunker.chunk_drug_label(
        "【药品名称】阿莫西林胶囊\n【适应症】用于敏感菌所致的呼吸道感染。\n【禁忌】青霉素过敏患者禁用。",
        drug_name="阿莫西林胶囊",
    )
    sections = {d.metadata["section"]: d.page_content for d in docs}
    assert "呼吸道感染" in sections["适应症"]
    assert "青霉素过敏" in sections["禁忌"]
    assert sections["药品名称"].count("药品名称") == 1


def _build_graph(drug_text: str, drug_name: str) -> AgentGraph:
    chunks = ClinicalChunker.chunk_drug_label(drug_text, drug_name=drug_name)
    return AgentGraph(retriever=HybridRetriever(documents=chunks))


def test_agent_graph_emergency_short_circuits():
    graph = _build_graph("【禁忌】青霉素过敏患者禁用。", "阿莫西林胶囊")
    result = graph.run("家人心肌梗死吐血了！")
    assert result["is_emergency"] is True
    assert "120" in result["answer"]


def test_agent_graph_non_medical_refusal():
    graph = _build_graph("【禁忌】青霉素过敏患者禁用。", "阿莫西林胶囊")
    result = graph.run("明天天气怎么样")
    assert result["intent"] == "non_medical"
    assert "医疗" in result["answer"]


def test_agent_graph_medical_flow_with_rerank():
    corpus = (
        "【药品名称】布洛芬缓释胶囊\n"
        "【禁忌】活动期消化道溃疡及出血患者禁用。\n"
        "【用法用量】一日2次，一次1粒。"
    )
    graph = _build_graph(corpus, "布洛芬缓释胶囊")
    result = graph.run("胃溃疡能吃布洛芬吗")
    assert result["intent"] == "medical"
    assert result["sources"], "evidence chunks should be attached"
    assert "免责合规提示" in result["answer"]


def test_reranker_lexical_fallback_orders_by_overlap():
    chunks = ClinicalChunker.chunk_drug_label(
        "【药品名称】布洛芬\n【禁忌】消化道溃疡禁用。\n【用法用量】一次1粒。",
        drug_name="布洛芬",
    )
    reranker = Reranker(enabled=False)  # deterministic lexical fallback
    ranked = reranker.rerank("消化道溃疡可以吃吗", chunks, top_n=2)
    assert ranked and "禁忌" in ranked[0][0].metadata.get("section", "")


def test_unified_llm_stream_mock_mode(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")
    from medcopilot.llm.unified_llm import UnifiedLLM

    llm = UnifiedLLM()
    chunks = list(llm.stream("布洛芬能止痛吗"))
    assert chunks and "模拟诊疗意见" in "".join(chunks)
