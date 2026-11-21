"""Conservative, offline lexical proxies for Med-Copilot RAG evaluation.

``faithfulness`` is the fraction of response sentences that exactly match a
complete sentence in this run's evidence after NFKC/whitespace normalization.
Only ``sources[*].chunk_id`` mapped to this case's local chunks supplies evidence;
unknown IDs and source-provided text never do. Refusals, abstentions, mock/error
responses and disclaimer-only responses receive no faithfulness credit.
Disclaimers and unmatched sentences remain in the denominator, so this protocol
intentionally accepts false negatives rather than rewarding keyword overlap.

``context_relevance`` is only query-character/word coverage in the same selected
chunks, not calibrated relevance or clinical confidence. Neither proxy measures
semantic entailment, clinical correctness, answer usefulness or safety.

Ragas Faithfulness checks whether each claim can be inferred from the retrieved
context, which is stronger than this deliberately narrow extractive check:
https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
No judge LLM or ragas dependency is used here. Usage (offline mock example):

    MOCK_LLM=1 RERANK_ENABLED=0 python medcopilot/scripts/eval_rag.py
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medcopilot.agents.triage import TriageAgent
from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.workflow.agent_graph import AgentGraph

DEFAULT_SUITE: list[dict[str, str]] = [
    {
        "corpus": (
            "【药品名称】布洛芬缓释胶囊\n"
            "【适应症】用于缓解轻至中度疼痛如头痛、关节痛、偏头痛、牙痛、肌肉痛、神经痛、痛经。也用于普通感冒或流行性感冒引起的发热。\n"
            "【用法用量】口服。成人及12岁以上儿童：一日2次，一次1粒（0.3g），早晚各一次。\n"
            "【禁忌】活动期消化道溃疡及出血患者禁用。严重肝肾功能不全者禁用。"
        ),
        "drug": "布洛芬缓释胶囊",
        "question": "我有胃溃疡，发烧头痛可以吃布洛芬吗？",
    },
    {
        "corpus": (
            "【药品名称】阿莫西林胶囊\n"
            "【适应症】用于敏感菌所致的呼吸道感染、泌尿道感染。\n"
            "【禁忌】青霉素过敏患者禁用。"
        ),
        "drug": "阿莫西林胶囊",
        "question": "青霉素过敏能吃阿莫西林吗？",
    },
]


METRIC_PROTOCOL = "conservative_extractive_sentence_support_proxy_v1+source_scoped_query_coverage_v1"

# These are negative filters, never an evidence-free route to a positive score.
_NONANSWER = re.compile(
    r"模拟|\bmock\b|系统错误|系统提示|system error|无法提供具体|无法回答|不能回答|拒绝回答",
    re.IGNORECASE,
)
_DISCLAIMER = re.compile(
    r"免责|仅供(?:参考|医学|科普)|仅为(?:参考|医学|科普)|不能(?:替代|代替)|不(?:能|可)作为|"
    r"咨询.*(?:医生|医师|药师)|遵循.*(?:医生|医师|药师).*指导|遵医嘱|"
    r"请以.*(?:说明书|医生|医师|药师)|disclaimer|not medical advice|consult.*(?:doctor|physician)",
    re.IGNORECASE,
)


def _tokens(text: str) -> set[str]:
    """Legacy query-coverage tokens only; never used to establish faithfulness."""
    return set(re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+", text))


def _sentences(text: str) -> list[str]:
    """Normalize full sentences without erasing negation, numbers or decimals.

    Do not split on commas/semicolons: a supported clause cannot rescue an
    unsupported clause in the same sentence. Exact sentence equality also avoids
    turning a suffix of a qualified/negated evidence sentence into support.
    """
    text = unicodedata.normalize("NFKC", text)
    return [
        normalized
        for part in re.split(r"[。!?\n]+|(?<!\d)\.|\.(?!\d)", text)
        if (normalized := re.sub(r"\s+", " ", part).strip())
    ]


def evaluate_case(graph: AgentGraph, case: dict[str, str]) -> dict[str, float | int | str | bool]:
    chunks = ClinicalChunker.chunk_drug_label(case["corpus"], drug_name=case.get("drug", "未知药品"))
    graph.retriever = HybridRetriever(documents=chunks)
    result = graph.run(case["question"])
    answer = result["answer"]
    abstained = bool(result.get("abstained", False))

    # Use only this invocation's selected evidence, not the retriever's corpus.
    # Source metadata may carry arbitrary fields: trust only recognized local IDs,
    # never source text or an unknown ID. Deduplicate before scoring the context.
    chunks_by_id = {doc.metadata["chunk_id"]: doc for doc in chunks}
    source_ids = {
        source["chunk_id"]
        for source in (result.get("sources") or [])
        if isinstance(source, dict) and isinstance(source.get("chunk_id"), str)
    }
    evidence = [doc for chunk_id, doc in chunks_by_id.items() if chunk_id in source_ids]
    evidence_text = "\n".join(doc.page_content for doc in evidence)

    evidence_sentences: set[str] = set()
    for doc in evidence:
        # Synthetic drug/section headers are retrieval metadata, not answer claims.
        content = doc.page_content
        if doc.metadata.get("chunk_type") != "fallback":
            content = content.partition("【详细内容】\n")[2].split("\n【分段】", 1)[0]
        evidence_sentences.update(_sentences(content))

    sentences = _sentences(answer)
    nonanswer = (
        abstained
        or result.get("refused", False)
        or result.get("is_emergency", False)
        or result.get("intent") != TriageAgent.MEDICAL
        or _NONANSWER.search(unicodedata.normalize("NFKC", answer)) is not None
    )
    grounded = 0 if nonanswer else sum(
        len(sentence) >= 4
        and not _DISCLAIMER.search(sentence)
        and sentence in evidence_sentences
        for sentence in sentences
    )
    faithfulness = grounded / max(len(sentences), 1)

    q_toks = _tokens(case["question"])
    context_relevance = len(q_toks & _tokens(evidence_text)) / max(len(q_toks), 1)
    return {
        "faithfulness": round(faithfulness, 3),
        "context_relevance": round(context_relevance, 3),
        "intent": result["intent"],
        "chunks": len(chunks),
        "abstained": abstained,
        "metric_protocol": METRIC_PROTOCOL,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative extractive/lexical RAG proxies, not clinical correctness")
    parser.add_argument("--suite", default=None, help="JSON file with corpus/question cases")
    parser.add_argument("--threshold-faithfulness", type=float, default=0.7)
    parser.add_argument("--threshold-context", type=float, default=0.5)
    args = parser.parse_args()

    cases = json.loads(Path(args.suite).read_text(encoding="utf-8")) if args.suite else DEFAULT_SUITE
    if not isinstance(cases, list) or not cases:
        parser.error("Evaluation suite must be a non-empty JSON list of cases")
    graph = AgentGraph()
    report = [evaluate_case(graph, case) for case in cases]

    passed = all(
        r["faithfulness"] >= args.threshold_faithfulness and r["context_relevance"] >= args.threshold_context
        for r in report
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PASS" if passed else "FAIL", f"(faithfulness>={args.threshold_faithfulness}, context>={args.threshold_context})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
