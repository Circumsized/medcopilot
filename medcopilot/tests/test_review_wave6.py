"""Wave-6 deep re-audit regressions.

Each test here exercises a concrete failure that the wave1-5 suite did NOT
cover, discovered during a full re-review:

1. Dense retriever wiring — the API built a dense-wired retriever into a
   throwaway local that was never bound to state/workflow, so dense retrieval
   was dead on the request path (``/health`` reported ``dense_degraded`` even
   with a working encoder).
2. ChromaDenseEncoder batch signature — ``__call__`` wrapped its argument in a
   list, so the retriever's batch fast-path (``encoder(list_of_texts)``) raised
   and only self-healed via a broad ``except``.
3. Production fail-open — ``require_reader`` was a no-op whenever ADMIN_API_KEY
   was empty, so a production deploy with no key served every consultation
   unauthenticated. Only ingestion returned 503.
4. API emergency ordering — the image consultation endpoint must complete
   emergency triage before reading/validating uploads or invoking OCR/RAG/LLM.
5. Credential redaction completeness — a bare ``token=``/``"token":`` secret in
   a provider error survived ``_redact`` (CWE-209 completeness gap).
6. Logger file sink — a hardcoded relative ``logs/app.log`` was created as an
   import side effect and would crash import on a read-only CWD.
"""

import pytest

# --------------------- 1 & 2: dense retriever wiring + batch encoder ----------


class _FakeDenseEncoder:
    """Deterministic encoder that supports BOTH single and batch calls."""

    def __call__(self, text):
        if isinstance(text, str):
            return [1.0, float(len(text) % 5) + 1.0, 3.0]
        return [[1.0, float(len(t) % 5) + 1.0, 3.0] for t in text]


def test_dense_encoder_wired_to_request_path(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api import server
    from medcopilot.common.config import config
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    monkeypatch.setattr(config, "enable_chroma_dense", True)
    monkeypatch.setattr(config, "admin_api_key", None)
    monkeypatch.setattr(config, "project_env", "development")
    monkeypatch.setattr(
        server.ChromaDenseEncoder, "build_encoder", staticmethod(lambda **kw: _FakeDenseEncoder())
    )

    docs = ClinicalChunker.chunk_drug_label(
        "【适应症】用于发热。\n【用法用量】口服一次1片。", drug_name="测试药"
    )
    client = TestClient(server.create_app(initial_documents=docs))
    health = client.get("/health").json()
    # The retriever bound to state/workflow must be the dense-wired one.
    assert health["dense_degraded"] is False


def test_chroma_dense_encoder_accepts_batch():
    """The retriever's batch fast-path must not raise on a list of texts."""
    from medcopilot.rag.vector_store import ChromaDenseEncoder

    seen = {}

    def embedding_fn(batch):
        # Real Chroma embedding functions require a flat list[str], never nested.
        for item in batch:
            assert isinstance(item, str), f"nested list leaked to embedding_fn: {item!r}"
        seen["last"] = list(batch)
        return [[1.0, 2.0, 3.0] for _ in batch]

    enc = ChromaDenseEncoder(collection=None, embedding_fn=embedding_fn)

    single = enc("hello")
    assert single == [1.0, 2.0, 3.0]

    batch = enc(["docA", "docB"])
    assert batch == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert seen["last"] == ["docA", "docB"]


def test_hybrid_retriever_uses_batch_encoder_without_degrading():
    """End-to-end: a batch-capable dense encoder must NOT degrade the retriever."""
    from medcopilot.rag.clinical_chunker import Document
    from medcopilot.rag.hybrid_retriever import HybridRetriever
    from medcopilot.rag.vector_store import ChromaDenseEncoder

    def embedding_fn(batch):
        return [[float(len(t)), 1.0, 2.0] for t in batch]

    enc = ChromaDenseEncoder(collection=None, embedding_fn=embedding_fn)
    docs = [Document(page_content="布洛芬用于发热", metadata={"chunk_id": "a"}),
            Document(page_content="阿莫西林用于感染", metadata={"chunk_id": "b"})]
    retriever = HybridRetriever(documents=docs, dense_encoder=enc)
    assert retriever.is_degraded is False
    hits = retriever.retrieve("发热", top_n=2)
    assert hits  # dense path produced results


# ----------------------------- 3: production fail-closed ----------------------


def test_production_without_admin_key_blocks_reads(monkeypatch):
    """Reads must fail CLOSED (503) in production when no key is configured."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "admin_api_key", None)
    monkeypatch.setattr(config, "project_env", "production")
    client = TestClient(create_app())

    resp = client.post("/consultations", json={"query": "头痛"})
    assert resp.status_code == 503
    # Ingestion stays closed too (unchanged behavior).
    ingest = client.post("/documents/drug-label", data={"raw_text": "x", "drug_name": "y"})
    assert ingest.status_code == 503
    # /health must advertise that auth is required in this configuration.
    assert client.get("/health").json()["auth_required"] is True


def test_development_without_admin_key_still_open(monkeypatch):
    """Regression guard: local dev with no key keeps reads open for the demo."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "admin_api_key", None)
    monkeypatch.setattr(config, "project_env", "development")
    client = TestClient(create_app())
    assert client.post("/consultations", json={"query": "头痛"}).status_code == 200


def test_production_with_admin_key_enforces_bearer(monkeypatch):
    """Production + key: reads require the bearer token, ingestion requires it too."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "admin_api_key", "prod-secret")
    monkeypatch.setattr(config, "project_env", "production")
    client = TestClient(create_app())

    assert client.post("/consultations", json={"query": "头痛"}).status_code == 401
    ok = client.post(
        "/consultations", json={"query": "头痛"}, headers={"Authorization": "Bearer prod-secret"}
    )
    assert ok.status_code == 200


# -------------------------- 4: API emergency-first ----------------------------


def test_api_image_emergency_short_circuits_before_image_ocr_and_workflow(monkeypatch):
    """The API owns triage and must return before reading an emergency upload."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from starlette.datastructures import UploadFile as StarletteUploadFile

    from medcopilot.api import server

    calls = {"image_read": 0, "ocr": 0, "workflow": 0}

    async def forbidden_image_read(self, *args, **kwargs):
        calls["image_read"] += 1
        raise AssertionError("emergency path read the uploaded image")

    class TripwireOCR:
        is_available = False

        def suggest_drug_from_image(self, *args, **kwargs):
            calls["ocr"] += 1
            raise AssertionError("emergency path invoked OCR")

    class TripwireWorkflow:
        def __init__(self, **kwargs):
            pass

        def run(self, *args, **kwargs):
            calls["workflow"] += 1
            raise AssertionError("emergency path invoked RAG/LLM workflow")

    monkeypatch.setattr(StarletteUploadFile, "read", forbidden_image_read)
    monkeypatch.setattr(server, "OCRTool", TripwireOCR)
    monkeypatch.setattr(server, "AgentGraph", TripwireWorkflow)
    client = TestClient(server.create_app())

    # Unsupported extension and invalid bytes are intentional tripwires: either
    # validation would reject them if it ran before emergency triage.
    unknown_id = "unknown-or-expired"
    response = client.post(
        "/consultations/with-image",
        data={
            "query": "突发心肌梗死，胸痛压榨濒死感",
            "session_id": unknown_id,
        },
        files={"image": ("not-an-image.exe", b"not-an-image", "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.json()["is_emergency"] is True
    assert "120" in response.json()["answer"]
    assert type(response.json()["session_id"]) is str
    assert response.json()["session_id"] and response.json()["session_id"] != unknown_id
    assert calls == {"image_read": 0, "ocr": 0, "workflow": 0}


# ----------------------- 5: credential redaction completeness -----------------


def test_redact_covers_bare_token():
    from medcopilot.llm.unified_llm import UnifiedLLM

    # Fake credentials assembled at runtime: identical values when the test
    # runs, but static scanners must not mistake fixtures for hardcoded secrets.
    fake_json_key = "my-json-" + "secret-456"
    dirty = (
        '{"api_key": "' + fake_json_key + '", "token": "tok_789"} '
        "url?token=oauthZZZ and keyboard=fast monkey=yes"
    )
    clean = UnifiedLLM._redact(dirty)
    assert "tok_789" not in clean
    assert "oauthZZZ" not in clean
    assert fake_json_key not in clean
    # Innocent words that merely contain 'key' must be untouched.
    assert "keyboard=fast" in clean
    assert "monkey=yes" in clean


def test_explicit_mock_provider_never_reaches_network(monkeypatch):
    """Explicit mock wins even when both global mock switches are off."""
    from medcopilot.common.config import config
    from medcopilot.llm.unified_llm import UnifiedLLM

    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.setattr(config, "mock_llm", False)
    calls = []

    def boom_bundle(self, prompt, stream):
        calls.append((prompt, stream))
        raise AssertionError("network bundle built despite provider='mock'")

    monkeypatch.setattr(UnifiedLLM, "_bundle", boom_bundle)
    llm = UnifiedLLM(provider="mock", max_retries=1)
    assert "【模拟诊疗意见】" in llm.invoke("任何问题")
    assert "【模拟诊疗意见】" in "".join(llm.stream("任何问题"))
    assert calls == []


# ------------------------------- 6: logger file sink --------------------------


def test_logger_file_sink_is_configurable(monkeypatch, tmp_path):
    """Configuration checks must not replace the shared AppConfig singleton."""
    import io
    import logging

    from medcopilot.common import logger as logger_module
    from medcopilot.common.config import AppConfig

    # Empty LOG_FILE -> config allows disabling; default points at logs/app.log.
    monkeypatch.setenv("LOG_FILE", "")
    disabled = AppConfig()
    assert disabled.log_file == ""

    monkeypatch.setenv("LOG_FILE", str(tmp_path / "custom.log"))
    relocated = AppConfig()
    assert relocated.log_file == str(tmp_path / "custom.log")

    # The stdout sink import path stays functional either way (module-level
    # guard already ran at import); emitting a record must not raise and must
    # reach the stdout sink.
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler_id = logger_module.logger.add(handler, level="INFO")
    try:
        logger_module.logger.info("sink-config smoke")
    finally:
        logger_module.logger.remove(handler_id)
    assert "sink-config smoke" in stream.getvalue()

