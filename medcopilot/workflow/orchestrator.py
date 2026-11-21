from __future__ import annotations

from typing import Any

from medcopilot.llm.unified_llm import UnifiedLLM
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.rag.reranker import Reranker
from medcopilot.workflow.agent_graph import AgentGraph, Session


class MedicalConsultationWorkflow:
    """Compatibility wrapper around AgentGraph.

    Existing callers can keep importing this class while API/UI and tests share
    the same rerank, evidence-floor, grounding, and safety behavior.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        llm: UnifiedLLM | None = None,
        reranker: Reranker | None = None,
        rerank_top_n: int | None = None,
        evidence_floor: float | None = None,
    ):
        self.graph = AgentGraph(
            retriever=retriever,
            llm=llm,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
            evidence_floor=evidence_floor,
        )

    @property
    def retriever(self) -> HybridRetriever:
        return self.graph.retriever

    @retriever.setter
    def retriever(self, value: HybridRetriever) -> None:
        self.graph.retriever = value

    @property
    def llm(self) -> UnifiedLLM:
        return self.graph.llm

    @llm.setter
    def llm(self, value: UnifiedLLM) -> None:
        self.graph.llm = value

    def run(
        self,
        user_query: str,
        drug_context: str | None = None,
        session: Session | None = None,
    ) -> dict[str, Any]:
        return self.graph.run(user_query, drug_context=drug_context, session=session)
