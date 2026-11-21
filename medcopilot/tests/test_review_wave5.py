"""Wave-5 deployment/security hardening regressions.

Covers: ASGI body-size middleware (pre-parse, incl. streamed bodies), bearer
auth (reader vs admin), with-image query length cap, MOCK_LLM AppConfig wiring
+ placeholder-key detection, package-level Session/ConsultationState exports,
UI session memory, and ingest thread-safety surface.
"""

import pytest

from medcopilot.workflow.safety_guard import MedicalSafetyGuard

# --------------------------------- auth ---------------------------------------

def test_auth_disabled_by_default_allows_all():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.post("/consultations", json={"query": "头痛"}).status_code == 200


def test_admin_key_required_for_ingest_when_configured(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "admin_api_key", "secret-admin-key")
    client = TestClient(create_app())
    payload = {"raw_text": "【药品名称】测试药\n【药理毒理】测试。", "drug_name": "测试药"}

    denied = client.post("/documents/drug-label", data=payload)
    assert denied.status_code == 401

    wrong = client.post(
        "/documents/drug-label", data=payload, headers={"Authorization": "Bearer wrong"}
    )
    assert wrong.status_code == 401

    ok = client.post(
        "/documents/drug-label", data=payload, headers={"Authorization": "Bearer secret-admin-key"}
    )
    assert ok.status_code == 200


def test_reader_key_enforced_on_consultations(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "admin_api_key", "secret-key")
    client = TestClient(create_app())

    denied = client.post("/consultations", json={"query": "头痛"})
    assert denied.status_code == 401

    ok = client.post(
        "/consultations", json={"query": "头痛"}, headers={"Authorization": "Bearer secret-key"}
    )
    assert ok.status_code == 200


def test_production_without_admin_key_disables_ingest(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "admin_api_key", None)
    monkeypatch.setattr(config, "project_env", "production")
    client = TestClient(create_app())
    resp = client.post("/documents/drug-label", data={"raw_text": "x", "drug_name": "y"})
    assert resp.status_code == 503


# ----------------------------- body size middleware ---------------------------

def test_body_size_middleware_rejects_oversized_content_length(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api import server

    monkeypatch.setattr(server, "MAX_BODY_BYTES", 64)
    client = TestClient(server.create_app())
    resp = client.post(
        "/consultations",
        json={"query": "头痛"},
        headers={"Content-Length": str(10 * 1024 * 1024)},
    )
    assert resp.status_code == 413


def test_body_size_middleware_rejects_streamed_oversized_body(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api import server

    monkeypatch.setattr(server, "MAX_BODY_BYTES", 32)
    client = TestClient(server.create_app())
    resp = client.post(
        "/consultations",
        json={"query": "x" * 2048},  # body far exceeds 32 bytes regardless of header
    )
    assert resp.status_code == 413


# ------------------------------ query length cap ------------------------------

def test_with_image_query_length_cap():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import MAX_QUERY_CHARS, create_app

    client = TestClient(create_app())
    resp = client.post(
        "/consultations/with-image",
        data={"query": "头" * (MAX_QUERY_CHARS + 1)},
        files={"image": ("box.jpg", b"\xff\xd8\xff" + b"0" * 32, "image/jpeg")},
    )
    assert resp.status_code == 422


# ------------------------------- mock wiring ----------------------------------

def test_mock_llm_reads_appconfig(monkeypatch):
    from medcopilot.common.config import config
    from medcopilot.llm.unified_llm import UnifiedLLM

    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setattr(config, "mock_llm", True)
    llm = UnifiedLLM(provider="openai_compatible", api_key="some-real-key", api_base="https://api.example.com/v1")
    assert llm._should_mock() is True
    monkeypatch.setattr(config, "mock_llm", False)
    assert llm._should_mock() is False


def test_placeholder_api_key_falls_back_to_mock(monkeypatch):
    from medcopilot.common.config import config
    from medcopilot.llm.unified_llm import UnifiedLLM

    # W2.1: the placeholder verdict moved into the single-source reason classifier
    # (a private ``_has_usable_key`` no longer exists). Deliberately clear the
    # mock switches: ``MOCK_LLM=1`` outranks the credential check and would
    # otherwise report reason="mock_env" before the placeholder is ever examined.
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setattr(config, "mock_llm", False)
    llm = UnifiedLLM(provider="openai_compatible", api_key="your-api-key-here", api_base="https://api.example.com/v1")
    assert llm._mock_reason() == "placeholder_key"
    assert llm._should_mock() is True


def test_real_key_not_mocked_for_remote(monkeypatch):
    from medcopilot.common.config import config
    from medcopilot.llm.unified_llm import UnifiedLLM

    monkeypatch.setenv("MOCK_LLM", "0")
    monkeypatch.setattr(config, "mock_llm", False)
    llm = UnifiedLLM(provider="openai_compatible", api_key="sk-real", api_base="https://api.example.com/v1")
    assert llm._should_mock() is False


# ----------------------------- package exports -------------------------------

def test_workflow_exports_session_and_state():
    from medcopilot.workflow import AgentGraph, ConsultationState, Session

    session = Session()
    session.record("q", "a", "medical", policy_version=MedicalSafetyGuard.POLICY_VERSION)
    assert session.turns[0]["query"] == "q"
    state = ConsultationState(query="x")
    assert state.query == "x"
    assert AgentGraph is not None


# ------------------------------- UI session ----------------------------------

def test_ui_build_app_exists_and_session_wired():
    gradio = pytest.importorskip("gradio")
    from medcopilot.ui import app as ui_app

    demo = ui_app.build_app()
    assert isinstance(demo, gradio.Blocks)


# --------------------------- ingest thread safety ----------------------------

def test_concurrent_ingest_no_duplicate_chunks():
    pytest.importorskip("fastapi")
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    client = TestClient(create_app())
    payload = {
        "raw_text": "【药品名称】并发测试药\n【禁忌】青霉素过敏患者禁用。\n【适应症】用于测试。",
        "drug_name": "并发测试药",
    }

    def do_ingest(_):
        return client.post("/documents/drug-label", data=payload).json()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(do_ingest, range(4)))

    sizes = [r["corpus_size"] for r in results]
    added = [r["chunks_added"] for r in results]
    # Exactly one request inserts the chunks; the rest see them already present.
    assert sorted(added) == [0, 0, 0, 3]
    assert len(set(sizes)) == 1, f"corpus size diverged across concurrent ingests: {sizes}"
