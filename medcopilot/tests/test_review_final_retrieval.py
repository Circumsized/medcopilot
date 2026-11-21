"""Query-time dense failures must never become a second lexical RRF vote."""

import pytest

from medcopilot.rag import hybrid_retriever as module
from medcopilot.rag.clinical_chunker import Document
from medcopilot.rag.hybrid_retriever import HybridRetriever


class QueryEncoder:
    def __init__(self, failure=None):
        self.failure = failure
        self.calls = 0

    def __call__(self, texts):
        return [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]][:len(texts)]

    def encode_query(self, query):
        self.calls += 1
        if self.failure == "exception":
            raise RuntimeError("query endpoint unavailable")
        if self.failure == "dimension":
            return [1.0, 0.0, 0.0]
        return [1.0, 0.0]


@pytest.fixture
def setup_retriever(monkeypatch):
    clock = [100.0]
    # W2.5:冷却时钟迁移到单调钟(NTP 跳变不得改变 30s 降级窗口),打桩同步迁移。
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    docs = [
        Document("pain pain", {"chunk_id": "a"}),
        Document("pain fever other words", {"chunk_id": "b"}),
        Document("infection", {"chunk_id": "c"}),
    ]
    encoder = QueryEncoder()
    return HybridRetriever(docs, dense_encoder=encoder), encoder, docs, clock


def assert_lexical_only(retriever, hits):
    lexical = retriever.bm25_search_with_scores("pain", top_n=3)
    assert [doc.page_content for doc, _ in hits] == [doc.page_content for doc, _ in lexical]
    assert [score for _, score in hits] == pytest.approx([score / lexical[0][1] for _, score in lexical])
    assert all(doc.metadata.get("retrieval_degraded") is True for doc, _ in hits)
    assert retriever._fallback_vectors is None  # fusion must not run TF-IDF at all


@pytest.mark.parametrize("failure", ["exception", "dimension"])
def test_query_failure_uses_only_bm25_and_recovers(setup_retriever, failure):
    retriever, encoder, docs, clock = setup_retriever
    assert not retriever.is_degraded
    encoder.failure = failure
    first = retriever.retrieve_with_scores("pain", top_n=3)
    assert_lexical_only(retriever, first)
    assert retriever.is_degraded
    assert encoder.calls == 1
    assert retriever._dense_encoder is encoder
    assert all("retrieval_degraded" not in doc.metadata for doc in docs)
    assert all(doc is not original for doc, _ in first for original in docs)

    encoder.failure = None
    clock[0] = 129.0
    assert_lexical_only(retriever, retriever.retrieve_with_scores("pain", top_n=3))
    assert encoder.calls == 1
    assert retriever.is_degraded

    clock[0] = 130.0
    recovered = retriever.retrieve_with_scores("pain", top_n=3)
    assert encoder.calls == 2
    assert not retriever.is_degraded
    assert len(recovered) == 3  # dense-only document restored
    assert all("retrieval_degraded" not in doc.metadata for doc, _ in recovered)
    assert all("retrieval_degraded" not in doc.metadata for doc in docs)
    assert all(doc.metadata["retrieval_degraded"] for doc, _ in first)


def test_success_uses_unchanged_two_list_rrf(setup_retriever):
    retriever, encoder, docs, _ = setup_retriever
    lexical = retriever.bm25_search("pain", top_n=6)
    expected = retriever.rrf_fuse_with_scores(lexical, [docs[1], docs[2]], top_n=3)
    assert retriever.retrieve_with_scores("pain", top_n=3) == expected
    assert encoder.calls == 1
    assert not retriever.is_degraded
    assert all("retrieval_degraded" not in doc.metadata for doc in docs)


@pytest.mark.parametrize("failure", ["exception", "dimension"])
def test_direct_dense_api_keeps_tfidf_during_failure_and_cooldown(setup_retriever, failure):
    retriever, encoder, docs, _ = setup_retriever
    encoder.failure = failure
    expected = HybridRetriever(docs).dense_search_with_scores("pain", top_n=3)
    assert retriever.dense_search_with_scores("pain", top_n=3) == expected
    assert retriever.is_degraded
    assert retriever.dense_search("pain", top_n=3) == [doc for doc, _ in expected]
    assert encoder.calls == 1
    assert all("retrieval_degraded" not in doc.metadata for doc in docs)


def test_no_encoder_does_not_mutate_source_metadata(setup_retriever):
    _, _, docs, _ = setup_retriever
    retriever = HybridRetriever(docs)
    assert_lexical_only(retriever, retriever.retrieve_with_scores("pain", top_n=3))
    assert all("retrieval_degraded" not in doc.metadata for doc in docs)


def test_overlapping_failure_does_not_override_successful_call(setup_retriever, monkeypatch):
    retriever, encoder, docs, _ = setup_retriever
    original = encoder.encode_query
    overlapping_hits = []

    def encode_with_overlap(query):
        if query == "pain":
            encoder.failure = "exception"
            overlapping_hits.extend(retriever.retrieve_with_scores("fever", top_n=3))
            encoder.failure = None
        return original(query)

    monkeypatch.setattr(encoder, "encode_query", encode_with_overlap)
    expected = retriever.rrf_fuse_with_scores(retriever.bm25_search("pain"), [docs[1], docs[2]], top_n=3)
    assert retriever.retrieve_with_scores("pain", top_n=3) == expected
    assert overlapping_hits
    assert all(doc.metadata.get("retrieval_degraded") for doc, _ in overlapping_hits)
    assert all("retrieval_degraded" not in doc.metadata for doc in docs)
