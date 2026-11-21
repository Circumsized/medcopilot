"""Wave 8 regression tests: runtime isolation + serving snapshots + auth tiering.

Pins batch-D mechanisms (each pins a concurrency/correctness property, not an
implementation detail):
1. LLM 并发隔离：状态随 ``LLMInvocation`` 返回值走——共享同一 UnifiedLLM 的
   两个并发请求，一方重试耗尽（llm:ConnectError）另一方成功，degradations
   精确互不串扰（替代原先实例级 ``_last_invocation_error`` 的覆盖竞态）。
2. AppRuntime 冻结快照：``commit_serving`` 后 holder.runtime 各字段同代；
   ingest/rebuild 后 serving 与 documents 同源（workflow.retriever is retriever）。
3. Session 只读视图：``turns`` 是 tuple 快照，外不可变；物理 3 轮截断语义不变。
4. ConsultationResult Mapping 契约：dict 兼容（__getitem__/get/dict/**解包）、
   未知键 KeyError、frozen 不可变、policy_version 在场。
5. /health 鉴权分级：匿名（配置密钥/生产无密钥）只见 liveness 核心，有效
   Bearer 见全量；匿名恒 200（Docker HEALTHCHECK 语义）。
6. 急救先于鉴权：命中急救红线的请求在鉴权之前返回 120 警示；急救路径
   llm/ocr 调用计数为 0（不回退到工作流）。
7. 413 裸 ASGI 直测：分块超限 / 谎报 content-length 的流式超限都恰好一次
   413（connection: close、body 非空）；小 body 不误杀。
8. TrustedHost 默认 "*" 全绿；白名单拒伪造 Host。
9. sanitize_log_value（CWE-117）与 sources[*].rerank_mode。
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from medcopilot.rag.clinical_chunker import ClinicalChunker, Document
from medcopilot.rag.reranker import Reranker
from medcopilot.workflow.agent_graph import AgentGraph, ConsultationResult, Session
from medcopilot.workflow.safety_guard import MedicalSafetyGuard

EMERGENCY_QUERY = "突发心肌梗死怎么办"
PLAIN_QUERY = "头痛如何治疗？"
SNAPSHOT_LABEL = "【药品名称】快照药品\n【用法用量】口服。一次1片，一日3次。\n【禁忌】过敏者禁用。\n"


class FixedRetriever:
    """Stub retriever returning fixed documents (stateless, concurrency-safe)."""

    def __init__(self, docs: list[Document]):
        self._docs = docs

    def retrieve_with_scores(self, query: str, top_n: int) -> list[tuple[Document, float]]:
        return [(doc, 0.9) for doc in self._docs]


class ExtractiveLLM:
    """Stub LLM implementing ONLY ``invoke`` — pins the agent-graph fallback
    branch for test doubles that predate the invoke_with_status contract."""

    def __init__(self, answer: str = "【用法用量】每日三次。"):
        self.answer = answer

    def invoke(self, prompt: str) -> str:
        return self.answer


def _graph(llm, *, sufficiency_enabled: bool = False) -> AgentGraph:
    docs = [Document("【用法用量】每日三次。", {"chunk_id": "a"})]
    return AgentGraph(
        retriever=FixedRetriever(docs),
        llm=llm,
        reranker=Reranker(enabled=False),
        sufficiency_enabled=sufficiency_enabled,
    )


# --------------------------- 1: LLM 并发隔离 ---------------------------------


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def post(self, *args, **kwargs):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"message": {"content": "第二次成功"}}]},
        )


def test_llm_mock_invocation_contract():
    from medcopilot.llm.unified_llm import LLMInvocation, UnifiedLLM

    llm = UnifiedLLM(provider="mock")
    invocation = llm.invoke_with_status("【患者主诉/提问】头痛")
    assert isinstance(invocation, LLMInvocation)
    assert invocation.mocked is True
    assert invocation.error_name is None
    assert invocation.text
    # invoke() 保持旧签名与返回类型：委托 invoke_with_status(...).text
    assert llm.invoke("问题") == llm.invoke_with_status("问题").text
    # frozen：状态随返回值走，实例上无可变状态可被并发改写
    with pytest.raises(dataclasses.FrozenInstanceError):
        invocation.error_name = "X"


def test_llm_concurrent_invocations_do_not_cross_contaminate(monkeypatch):
    """One shared UnifiedLLM, two threads: the failing request must report its
    own outage and the successful one must stay clean — for every interleaving."""
    from medcopilot.llm import unified_llm as llm_module
    from medcopilot.llm.unified_llm import UnifiedLLM

    graph = _graph(UnifiedLLM(max_retries=1, retry_backoff=0))
    thread_local = threading.local()

    def bundler(self, prompt, stream=False):
        if getattr(thread_local, "fail", False):
            raise httpx.ConnectError("provider down")
        return {"url": "http://unit.test/v1", "headers": {}, "payload": {}, "params": None}

    monkeypatch.setattr(UnifiedLLM, "_should_mock", lambda self: False)
    monkeypatch.setattr(UnifiedLLM, "_bundle", bundler)
    monkeypatch.setattr(llm_module.httpx, "Client", _FakeClient)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)

    def execute_round(fail_idx: int) -> dict[int, ConsultationResult]:
        barrier = threading.Barrier(2)
        outcomes: dict[int, ConsultationResult] = {}

        def worker(idx: int) -> None:
            thread_local.fail = idx == fail_idx
            barrier.wait()  # both requests in flight simultaneously
            outcomes[idx] = graph.run("如何用药？")

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, i) for i in (0, 1)]
            for future in futures:
                future.result()
        return outcomes

    for round_index in range(20):  # ≥20 interleavings, alternating the loser
        loser_idx = round_index % 2
        outcomes = execute_round(loser_idx)
        assert outcomes[loser_idx]["degradations"] == ["llm:ConnectError"]
        assert outcomes[1 - loser_idx]["degradations"] == []


# --------------------------- 2: serving snapshots -----------------------------


def test_commit_serving_publishes_coherent_generation(monkeypatch):
    from medcopilot.api.runtime import AppRuntime, ServingHolder, commit_serving

    workflow = SimpleNamespace(retriever="r0")
    holder = ServingHolder(
        AppRuntime(
            documents=[], retriever="r0", dense_encoder=None, ocr="ocr", workflow=workflow, audit="audit"
        )
    )
    new_retriever = SimpleNamespace(name="r1")
    # Spy on the commit point: the graph must see the new retriever BEFORE the
    # holder publishes the new snapshot. Asserting after commit_serving returns
    # cannot distinguish "assign-then-swap" from "swap-then-assign" (both end
    # coherent), so the ordering check has to live inside the commit call.
    observed: dict[str, object] = {}
    real_swap = ServingHolder.swap

    def spy_swap(self, runtime):
        observed["retriever_at_swap"] = workflow.retriever
        observed["holder_before_swap"] = self.runtime.retriever
        real_swap(self, runtime)

    monkeypatch.setattr(ServingHolder, "swap", spy_swap)
    runtime = commit_serving(
        holder,
        documents=[1, 2],
        retriever=new_retriever,
        dense_encoder="enc",
        ocr="ocr",
        workflow=workflow,
        audit="audit",
    )
    assert holder.runtime is runtime
    assert runtime.retriever is new_retriever
    # commit point order enforced at swap time, not after the fact
    assert observed["retriever_at_swap"] is new_retriever
    assert observed["holder_before_swap"] == "r0"
    assert workflow.retriever is new_retriever
    assert runtime.documents == [1, 2]
    assert runtime.dense_encoder == "enc"
    assert runtime.ocr == "ocr" and runtime.audit == "audit"


def test_ingest_publishes_coherent_serving_snapshot():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import medcopilot.api.server as server_module

    seed = ClinicalChunker.chunk_drug_label(
        "【药品名称】种子药品\n【用法用量】口服。", drug_name="种子药品"
    )
    app = server_module.create_app(initial_documents=list(seed))
    client = TestClient(app)
    resp = client.post(
        "/documents/drug-label", data={"raw_text": SNAPSHOT_LABEL, "drug_name": "快照药品"}
    )
    assert resp.status_code == 200
    runtime = app.state.runtime_holder.runtime
    # One generation: the graph's retriever IS the serving retriever, built
    # from the same document snapshot the endpoint reports.
    assert runtime.workflow.retriever is runtime.retriever
    assert [d.metadata.get("drug_name") for d in runtime.retriever.documents].count("快照药品") >= 1
    assert any(d.metadata.get("drug_name") == "快照药品" for d in runtime.documents)
    assert resp.json()["corpus_size"] == len(runtime.documents)


class _StubEncoder:
    """encoder 协议最小实现（不依赖 chromadb）。"""

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        self.docs: list = []

    def index_documents(self, documents) -> int:
        self.docs = list(documents)
        return len(self.docs)

    def count(self) -> int:
        return len(self.docs)

    def __call__(self, text):
        if isinstance(text, list):
            # 仅支持单条编码：批量尝试抛错，走 HybridRetriever 的逐条回退
            raise TypeError("per-document encoder only")
        return [0.0, 0.0, 0.0]  # 固定维度向量，避免 HybridRetriever 判定降级丢弃 encoder


def test_rebuild_swaps_serving_snapshot(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import medcopilot.api.server as server_module

    configured = SimpleNamespace(**server_module.config.model_dump())
    configured.med_corpus_path = str(tmp_path / "corpus.jsonl")
    configured.enable_chroma_dense = True
    monkeypatch.setattr(server_module, "config", configured)
    app = server_module.create_app(dense_encoder_factory=lambda name: _StubEncoder(name))
    client = TestClient(app)
    assert (
        client.post(
            "/documents/drug-label", data={"raw_text": SNAPSHOT_LABEL, "drug_name": "快照药品"}
        ).status_code
        == 200
    )
    assert client.post("/index/rebuild").status_code == 200
    runtime = app.state.runtime_holder.runtime
    assert runtime.workflow.retriever is runtime.retriever
    assert runtime.retriever._dense_encoder is runtime.dense_encoder
    assert client.get("/index/status").json()["corpus_matches_index"] is True


# --------------------------- 3: Session read-only view ------------------------


def test_session_turns_is_read_only_tuple_with_physical_cap():
    session = Session()
    for i in range(10):
        session.record(f"q{i}", f"a{i}", "medical", policy_version=MedicalSafetyGuard.POLICY_VERSION)
    assert isinstance(session.turns, tuple)
    # 既有只读用法零改动：索引 / len / 迭代
    assert len(session.turns) == Session.MAX_RECORDED_TURNS
    assert session.turns[-1]["query"] == "q9"
    assert [t["query"] for t in session.turns] == ["q7", "q8", "q9"]
    assert session.turns[0]["policy_version"] == str(MedicalSafetyGuard.POLICY_VERSION)
    # 外部不可 append / 不可重绑
    with pytest.raises(AttributeError):
        session.turns.append({"query": "x"})
    with pytest.raises(AttributeError):
        session.turns = []


# --------------------------- 4: ConsultationResult Mapping -------------------


RESULT_KEYS = {
    "answer", "sources", "is_emergency", "intent",
    "abstained", "degradations", "policy_version", "dosage_gate",
}


def test_consultation_result_mapping_contract():
    result = _graph(ExtractiveLLM()).run("如何用药？")
    assert isinstance(result, ConsultationResult)
    # dict 兼容语义
    assert set(result) == RESULT_KEYS
    assert result["abstained"] is False
    assert result.get("abstained") is False
    assert result.get("nope") is None
    assert dict(result)["policy_version"] == MedicalSafetyGuard.POLICY_VERSION
    assert "policy_version" in result
    # 未知键 KeyError
    with pytest.raises(KeyError):
        result["nope"]
    # frozen 不可变
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.abstained = True
    # **result 解包进 ConsultationResponse 与逐字段构造等价
    pytest.importorskip("fastapi")
    from medcopilot.api.server import ConsultationResponse

    response = ConsultationResponse(**{**result, "session_id": "sid"})
    explicit = ConsultationResponse(
        answer=result["answer"],
        sources=result["sources"],
        is_emergency=result["is_emergency"],
        intent=result["intent"],
        abstained=result["abstained"],
        degradations=result["degradations"],
        policy_version=result["policy_version"],
        dosage_gate=result["dosage_gate"],
        session_id="sid",
    )
    assert response.model_dump() == explicit.model_dump()


def test_sources_carry_rerank_mode():
    result = _graph(ExtractiveLLM()).run("如何用药？")
    assert result["sources"], "expected at least one source entry"
    for source in result["sources"]:
        assert source["rerank_mode"] == "disabled"


# --------------------------- 5: /health auth tiering -------------------------


def test_health_anonymous_with_key_returns_core_payload_only(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "admin_api_key", "secret-key")
    monkeypatch.setattr(config, "project_env", "development")
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["auth_required"] is True
    # 运营细节不再匿名可见（OWASP API8）
    for leaked in (
        "corpus_documents", "index_stale", "index_manifest",
        "rerank_mode", "dense_degraded", "ocr_available",
    ):
        assert leaked not in body
    # 误配可见信号留在核心载荷
    assert "env_class" in body and "env_recognized" in body


def test_health_with_valid_bearer_returns_full_payload(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "admin_api_key", "secret-key")
    monkeypatch.setattr(config, "project_env", "development")
    client = TestClient(create_app())
    body = client.get("/health", headers={"Authorization": "Bearer secret-key"}).json()
    for expected in (
        "corpus_documents", "index_stale", "index_manifest",
        "rerank_mode", "dense_degraded", "ocr_available",
    ):
        assert expected in body


def test_health_production_without_key_stays_200_liveness(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "production")
    monkeypatch.setattr(config, "admin_api_key", None)
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["auth_required"] is True


# --------------------------- 6: emergency precedes auth ----------------------


def test_emergency_precedes_auth_production_without_key(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "production")
    monkeypatch.setattr(config, "admin_api_key", None)
    client = TestClient(create_app())
    resp = client.post("/consultations", json={"query": EMERGENCY_QUERY})
    assert resp.status_code == 200
    assert resp.json()["is_emergency"] is True


def test_production_without_key_ordinary_query_fails_closed(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "production")
    monkeypatch.setattr(config, "admin_api_key", None)
    client = TestClient(create_app())
    assert client.post("/consultations", json={"query": PLAIN_QUERY}).status_code == 503


def test_with_key_without_bearer_emergency_200_ordinary_401(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "development")
    monkeypatch.setattr(config, "admin_api_key", "secret-key")
    client = TestClient(create_app())
    emergency = client.post("/consultations", json={"query": EMERGENCY_QUERY})
    assert emergency.status_code == 200
    assert emergency.json()["is_emergency"] is True
    ordinary = client.post("/consultations", json={"query": PLAIN_QUERY})
    assert ordinary.status_code == 401


def test_emergency_path_never_invokes_llm_or_ocr(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import medcopilot.api.server as server_module
    from medcopilot.common.config import config

    calls = {"run": 0, "ocr": 0}

    class CountingGraph(server_module.AgentGraph):
        def run(self, *args, **kwargs):
            calls["run"] += 1
            return super().run(*args, **kwargs)

    original_ocr = server_module.OCRTool.suggest_drug_from_image

    def counting_ocr(self, *args, **kwargs):
        calls["ocr"] += 1
        return original_ocr(self, *args, **kwargs)

    monkeypatch.setattr(server_module, "AgentGraph", CountingGraph)
    monkeypatch.setattr(server_module.OCRTool, "suggest_drug_from_image", counting_ocr)
    monkeypatch.setattr(config, "project_env", "development")
    monkeypatch.setattr(config, "admin_api_key", "secret-key")
    client = TestClient(server_module.create_app())
    resp = client.post("/consultations", json={"query": EMERGENCY_QUERY})
    assert resp.status_code == 200
    assert resp.json()["is_emergency"] is True
    assert calls == {"run": 0, "ocr": 0}


# ------------------- 6b: triage precedes schema validation (B-1) ---------------


def _jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff" + b"0" * 32


def test_with_image_emergency_precedes_auth_production_without_key(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "production")
    monkeypatch.setattr(config, "admin_api_key", None)
    client = TestClient(create_app())
    resp = client.post(
        "/consultations/with-image",
        data={"query": EMERGENCY_QUERY},
        files={"image": ("box.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["is_emergency"] is True


def test_with_image_ordinary_query_fails_closed_production(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "production")
    monkeypatch.setattr(config, "admin_api_key", None)
    client = TestClient(create_app())
    resp = client.post(
        "/consultations/with-image",
        data={"query": PLAIN_QUERY},
        files={"image": ("box.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 503


def test_overlong_emergency_query_still_gets_120_answer(monkeypatch):
    """Pydantic-level length rejection must not preempt the 120 red-line.

    A pasted long history that mentions an emergency used to fail 422 before
    the handler ever ran the triage.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import MAX_QUERY_CHARS, create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "production")
    monkeypatch.setattr(config, "admin_api_key", None)
    client = TestClient(create_app())
    # Far past MAX_QUERY_CHARS, with the red-flag phrase in the MIDDLE: the scan
    # is full-text, so triage must fire before the length rejection (an earlier
    # head+tail scan window silently missed mid-text emergencies).
    padding = "既往病史" * ((MAX_QUERY_CHARS // 4) + 100)
    long_query = padding[: len(padding) // 2] + EMERGENCY_QUERY + padding[len(padding) // 2 :]
    assert len(long_query) > MAX_QUERY_CHARS
    resp = client.post("/consultations", json={"query": long_query})
    assert resp.status_code == 200
    assert resp.json()["is_emergency"] is True


def test_with_image_missing_image_field_still_gets_120_answer(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "production")
    monkeypatch.setattr(config, "admin_api_key", None)
    client = TestClient(create_app())
    # No `files` at all: FastAPI used to reject this with 422 (missing field)
    # before any handler code could run the triage.
    resp = client.post("/consultations/with-image", data={"query": EMERGENCY_QUERY})
    assert resp.status_code == 200
    assert resp.json()["is_emergency"] is True


def test_overlong_non_emergency_query_still_422(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import MAX_QUERY_CHARS, create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "development")
    monkeypatch.setattr(config, "admin_api_key", "secret-key")
    client = TestClient(create_app())
    resp = client.post(
        "/consultations",
        json={"query": "头" * (MAX_QUERY_CHARS + 1)},
        headers={"Authorization": "Bearer secret-key"},
    )
    assert resp.status_code == 422
    assert "too long" in resp.json()["detail"]


def test_emergency_answer_carries_120_guidance(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "project_env", "development")
    monkeypatch.setattr(config, "admin_api_key", None)
    client = TestClient(create_app())
    body = client.post("/consultations", json={"query": EMERGENCY_QUERY}).json()
    assert "120" in body["answer"]


def test_health_invalid_bearer_returns_core_payload_not_401(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "admin_api_key", "secret-key")
    monkeypatch.setattr(config, "project_env", "development")
    client = TestClient(create_app())
    # A wrong bearer must degrade to the liveness core, never 401 and never the
    # full operational payload — the same rule the anonymous path follows.
    resp = client.get("/health", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["auth_required"] is True
    for leaked in ("corpus_documents", "index_stale", "rerank_mode"):
        assert leaked not in body


# --------------------------- 7: 413 raw ASGI ---------------------------------


def _raw_asgi(app, scope: dict, receive) -> list[dict]:
    sent: list[dict] = []

    async def send(message: dict) -> None:
        sent.append(message)

    async def drive() -> None:
        await app(scope, receive, send)

    asyncio.run(drive())
    return sent


def _scope(method: str = "POST", headers: list | None = None) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": "/consultations",
        "raw_path": b"/consultations",
        "query_string": b"",
        "root_path": "",
        "headers": headers or [(b"host", b"testserver")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }


def _body_receive(chunks: list[dict]):
    queue = list(chunks)

    async def receive() -> dict:
        if queue:
            return queue.pop(0)
        return {"type": "http.disconnect"}

    return receive


def _single_413_assertions(sent: list[dict]) -> None:
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1, f"expected exactly one response.start, got {len(starts)}"
    assert starts[0]["status"] == 413
    headers = dict(starts[0]["headers"])
    assert headers.get(b"connection") == b"close"
    bodies = [m for m in sent if m["type"] == "http.response.body"]
    assert bodies and bodies[0]["body"]
    assert b"(streamed)" in bodies[0]["body"]


def test_raw_asgi_chunked_oversize_rejected_once(monkeypatch):
    pytest.importorskip("fastapi")
    import medcopilot.api.server as server_module

    monkeypatch.setattr(server_module, "MAX_BODY_BYTES", 32)  # 必须在 create_app() 之前
    app = server_module.create_app()
    headers = [(b"host", b"testserver"), (b"content-type", b"application/json")]
    sent = _raw_asgi(app, _scope(headers=headers), _body_receive([
        {"type": "http.request", "body": b"x" * 64, "more_body": False},
    ]))
    _single_413_assertions(sent)


def test_raw_asgi_lying_content_length_streamed_oversize_rejected_once(monkeypatch):
    pytest.importorskip("fastapi")
    import medcopilot.api.server as server_module

    monkeypatch.setattr(server_module, "MAX_BODY_BYTES", 32)  # 必须在 create_app() 之前
    app = server_module.create_app()
    headers = [
        (b"host", b"testserver"),
        (b"content-type", b"application/json"),
        (b"content-length", b"4"),  # 谎报小体积，实际流式超限
    ]
    sent = _raw_asgi(app, _scope(headers=headers), _body_receive([
        {"type": "http.request", "body": b"x" * 16, "more_body": True},
        {"type": "http.request", "body": b"x" * 16, "more_body": True},
        {"type": "http.request", "body": b"x" * 16, "more_body": False},
    ]))
    _single_413_assertions(sent)


def test_raw_asgi_small_body_is_not_rejected(monkeypatch):
    pytest.importorskip("fastapi")
    import medcopilot.api.server as server_module

    monkeypatch.setattr(server_module, "MAX_BODY_BYTES", 32)  # 必须在 create_app() 之前
    app = server_module.create_app()
    headers = [(b"host", b"testserver"), (b"content-type", b"application/json")]
    sent = _raw_asgi(app, _scope(headers=headers), _body_receive([
        {"type": "http.request", "body": b"", "more_body": False},
    ]))
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] != 413  # 负向防误杀：小 body 走正常校验路径（422）


# --------------------------- 8: TrustedHost ----------------------------------


def test_trusted_hosts_default_wildcard_allows_all():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import medcopilot.api.server as server_module

    client = TestClient(server_module.create_app())
    assert client.get("/health").status_code == 200


def test_trusted_hosts_whitelist_rejects_foreign_host(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import medcopilot.api.server as server_module
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "trusted_hosts", "med.example")
    app = server_module.create_app()
    client = TestClient(app, base_url="http://med.example")
    assert client.get("/health").status_code == 200
    # 伪造 Host：手工构造 scope 直连 ASGI 栈
    sent = _raw_asgi(
        app,
        _scope(method="GET", headers=[(b"host", b"evil.example")]),
        _body_receive([]),
    )
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1
    assert starts[0]["status"] == 400


# --------------------------- 9: log sanitization -----------------------------


def test_sanitize_log_value_escapes_control_characters():
    from medcopilot.api.server import sanitize_log_value

    cleaned = sanitize_log_value("药名\n第二行\t结尾\r\x01")
    assert "\n" not in cleaned and "\t" not in cleaned and "\r" not in cleaned
    assert "\x01" not in cleaned
    assert "\\n" in cleaned and "\\t" in cleaned and "\\r" in cleaned and "\\x01" in cleaned
    assert sanitize_log_value("正常药名") == "正常药名"


def test_sanitize_log_value_truncates():
    from medcopilot.api.server import sanitize_log_value

    assert sanitize_log_value("a" * 500, limit=200) == "a" * 200
    assert sanitize_log_value("药" * 500) == "药" * 200
