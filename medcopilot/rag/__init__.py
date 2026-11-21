"""RAG package surface.

``ChromaDenseEncoder`` is exposed lazily (PEP 562) on purpose. Importing
``medcopilot.rag.vector_store`` registers the embedding function with
chromadb, which means it imports chromadb — and the ingest-validation path is
pinned by ``tests/test_ingest_validation.py`` to stay free of vector-store and
model imports so it remains testable offline with no optional extras. A
package-level re-export would drag chromadb into every ``rag`` submodule import,
so the heavy name is resolved only when it is actually asked for.
"""

from medcopilot.rag.clinical_chunker import ClinicalChunker, Document
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.rag.reranker import Reranker

__all__ = ["ChromaDenseEncoder", "ClinicalChunker", "Document", "HybridRetriever", "Reranker"]


def __getattr__(name: str):
    if name == "ChromaDenseEncoder":
        from medcopilot.rag.vector_store import ChromaDenseEncoder

        return ChromaDenseEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
