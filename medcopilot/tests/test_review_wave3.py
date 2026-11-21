"""Wave 3 regression and hardening tests.

Pins fixes for:
- HybridRetriever: graceful degradation, dimension mismatch, NaN/Inf rejection, query cooldown
- UnifiedLLM: credential redaction (URL query, Basic, Bearer with Base64, JSON body)
- UnifiedLLM: streaming error safety (no fake mock appended after real chunks yielded)
- ClinicalChunker: drug-aware fallback chunk_id and preamble retention
- API / Server: CORS preflight on the strict-allowlist branch (the wildcard
  cors_allow_all branch is pinned in wave7), emergency pre-filtering before OCR
- Workflow: dynamic exports without circular import
"""

import httpx
import pytest

from medcopilot.llm.unified_llm import UnifiedLLM
from medcopilot.rag.clinical_chunker import ClinicalChunker, Document
from medcopilot.rag.hybrid_retriever import HybridRetriever

# ----------------------------- HybridRetriever ------------------------------

def _sample_docs():
    return [
        Document(
            page_content="布洛芬缓释胶囊用于缓解轻至中度疼痛，如头痛、关节痛、偏头痛、牙痛。",
            metadata={"drug_name": "布洛芬", "section": "适应症", "chunk_id": "c1"},
        ),
        Document(
            page_content="活动期消化道溃疡患者禁用布洛芬。严重心力衰竭及孕妇禁用。",
            metadata={"drug_name": "布洛芬", "section": "禁忌", "chunk_id": "c2"},
        ),
        Document(
            page_content="对阿司匹林或其他非甾体抗炎药过敏者禁用布洛芬。",
            metadata={"drug_name": "布洛芬", "section": "注意事项", "chunk_id": "c3"},
        ),
    ]


def test_retriever_dense_init_failure_falls_back_gracefully():
    def failing_encoder(_texts):
        raise RuntimeError("network unreachable: cannot download BGE-M3 weights")

    retriever = HybridRetriever(documents=_sample_docs(), dense_encoder=failing_encoder)
    assert retriever.is_degraded is True
    results = retriever.retrieve("消化道溃疡可以吃布洛芬吗", top_n=2)
    assert len(results) > 0
    assert any("禁忌" in doc.metadata.get("section", "") for doc in results)


def test_retriever_query_dimension_mismatch_falls_back():
    class DimMismatchEncoder:
        def __call__(self, texts):
            return [[0.1] * 8 for _ in texts]

        def encode_query(self, _query):
            return [0.1] * 16  # mismatched query dim

    retriever = HybridRetriever(documents=_sample_docs(), dense_encoder=DimMismatchEncoder())
    assert retriever.is_degraded is False
    results = retriever.retrieve("头痛怎么办", top_n=2)
    assert len(results) > 0


def test_retriever_nan_inf_dense_vectors_trigger_fallback():
    class NanEncoder:
        def __call__(self, texts):
            return [[float("nan")] * 4 for _ in texts]

    retriever = HybridRetriever(documents=_sample_docs(), dense_encoder=NanEncoder())
    assert retriever.is_degraded is True


def test_retriever_cooldown_prevents_repeated_slow_retries():
    call_count = 0

    class TransientFailEncoder:
        def __call__(self, texts):
            return [[0.1, 0.2] for _ in texts]

        def encode_query(self, _query):
            nonlocal call_count
            call_count += 1
            raise httpx.ConnectTimeout("dense endpoint timeout")

    retriever = HybridRetriever(documents=_sample_docs(), dense_encoder=TransientFailEncoder())
    # First query fails and enters cooldown
    retriever.retrieve("头痛", top_n=1)
    assert call_count == 1
    # Second query within cooldown should NOT invoke encode_query again
    retriever.retrieve("牙痛", top_n=1)
    assert call_count == 1


# -------------------------------- UnifiedLLM --------------------------------

def test_llm_error_message_redacts_access_token(monkeypatch):
    from medcopilot.common.config import config

    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.setattr(config, "mock_llm", False)
    llm = UnifiedLLM(
        provider="openai_compatible", api_key="sk-test", api_base="https://api.example.com/v1",
        max_retries=1,
    )

    def raising_bundle(self, prompt, stream):
        req = httpx.Request(
            "POST",
            "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro?access_token=SECRET_TOKEN_123",
        )
        resp = httpx.Response(401, request=req, content=b'{"error_msg":"token expired"}')
        resp.raise_for_status()

    llm._bundle = raising_bundle.__get__(llm, UnifiedLLM)
    answer = llm.invoke("hello")
    assert "SECRET_TOKEN_123" not in answer
    assert "access_token=***" in answer


def test_llm_redact_comprehensive():
    # Fake credentials assembled at runtime: identical values when the test
    # runs, but static scanners must not mistake fixtures for hardcoded secrets.
    fake_bearer = "sk-live_" + "ABC123+xyz/99="
    fake_json_key = "my-json-" + "secret-456"
    dirty = (
        "POST https://api.openai.com/v1/chat/completions "
        "Authorization: Bearer " + fake_bearer + " "
        '{"api_key": "' + fake_json_key + '", "token": "tok_789"} '
        "url?client_secret=topsecret&access_token=oauth999 "
        "and normal words like monkey=yes keyboard=fast"
    )
    clean = UnifiedLLM._redact(dirty)
    assert fake_bearer not in clean
    assert fake_json_key not in clean
    assert "topsecret" not in clean
    assert "oauth999" not in clean
    # Ensure innocent words are untouched
    assert "monkey=yes" in clean
    assert "keyboard=fast" in clean


def test_llm_streaming_partial_failure_does_not_append_mock(monkeypatch):
    """Exercise the actual streaming implementation with an interrupted transport."""
    from medcopilot.common.config import config

    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.setattr(config, "mock_llm", False)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield 'data: {"choices": []}'
            yield 'data: {"choices": [{"delta": {"content": "根据说明书第一条，"}}]}'
            raise httpx.ReadTimeout("mid-stream disconnect")

    class Client(Response):
        def stream(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: Client())
    llm = UnifiedLLM(provider="openai_compatible", api_key="sk-test", api_base="https://api.example.com/v1")
    chunks = list(llm.stream("头痛咨询"))
    assert chunks[0] == "根据说明书第一条，"
    assert "流式传输已中断" in "".join(chunks)
    assert "模拟诊疗意见" not in "".join(chunks)


# ------------------------------ Chunker metadata ------------------------------

def test_chunker_fallback_chunk_differentiates_drugs():
    """Different drugs with identical text must NOT collide on chunk_id."""
    text = "本品性状为白色薄膜衣片。"
    doc_a = ClinicalChunker.chunk_drug_label(text, drug_name="阿司匹林")[0]
    doc_b = ClinicalChunker.chunk_drug_label(text, drug_name="布洛芬")[0]
    assert doc_a.metadata["chunk_id"] != doc_b.metadata["chunk_id"]


def test_chunker_retains_preamble():
    raw = "通用名称：布洛芬片。化学名为2-(4-异丁基苯基)丙酸。\n【适应症】用于头痛。"
    docs = ClinicalChunker.chunk_drug_label(raw, drug_name="布洛芬")
    sections = [d.metadata["section"] for d in docs]
    assert "概述前言" in sections
    assert "适应症" in sections


# --------------------------------- API hardening ------------------------------

def test_cors_preflight_blocks_unknown_origin():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())
    resp = client.options(
        "/consultations",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert resp.status_code == 400
    allowed = client.options(
        "/consultations",
        headers={"Origin": "http://localhost:7860", "Access-Control-Request-Method": "POST"},
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:7860"


def test_consult_with_image_emergency_short_circuits():
    """Emergency query must be intercepted in milliseconds without processing file."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())
    resp = client.post(
        "/consultations/with-image",
        data={"query": "突发剧烈胸痛压榨感濒死感两小时"},
        files={"image": ("drug.jpg", b"fake-content", "image/jpeg")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["is_emergency"] is True
    assert "120" in payload["answer"]


def test_workflow_package_lazy_exports():
    from medcopilot.workflow import AgentGraph, MedicalSafetyGuard
    assert AgentGraph is not None
    assert MedicalSafetyGuard is not None
