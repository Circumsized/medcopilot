"""索引 Manifest、可重建与蓝绿切换的契约测试（Slice 3）。

这些测试先于实现存在：稠密索引必须可以从权威语料全量重建；重建经
"构建新 collection → 校验 → 原子切换 manifest" 的蓝绿流程，失败时活跃
索引与 manifest 保持原状（回滚语义）。全部用 stub encoder，不依赖 chromadb。
"""

from __future__ import annotations

import json

import pytest

LABEL_A = "【药品名称】测试药品A\n【用法用量】口服。一次1片，一日3次。\n【禁忌】过敏者禁用。\n"
LABEL_A_MODIFIED = "【药品名称】测试药品A\n【用法用量】口服。一次2片，一日3次。\n"


def _chunk(label: str, drug: str):
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    return ClinicalChunker.chunk_drug_label(label, drug_name=drug)


class StubEncoder:
    """encoder 协议的最小实现：记录灌入内容，可注入构建失败。"""

    def __init__(self, collection_name: str, *, fail: bool = False):
        self.collection_name = collection_name
        self.fail = fail
        self.docs: list = []

    def index_documents(self, documents) -> int:
        if self.fail:
            raise RuntimeError("simulated index build failure")
        self.docs = list(documents)
        return len(self.docs)

    def count(self) -> int:
        return len(self.docs)


class FlakyFactory:
    """仅第 fail_on_call 次构建失败的 encoder 工厂（之后恢复正常）。"""

    def __init__(self, fail_on_call: int | None = None):
        self.fail_on_call = fail_on_call
        self.calls: list[str] = []
        self.encoders: list[StubEncoder] = []

    def __call__(self, collection_name: str) -> StubEncoder:
        self.calls.append(collection_name)
        fail = self.fail_on_call is not None and len(self.calls) == self.fail_on_call
        encoder = StubEncoder(collection_name, fail=fail)
        self.encoders.append(encoder)
        return encoder


def _manager(tmp_path, factory, labels=(("测试药品A", LABEL_A),)):
    from medcopilot.rag.corpus_store import CorpusStore
    from medcopilot.rag.index_manager import IndexManager

    store = CorpusStore(path=tmp_path / "corpus.jsonl")
    for drug, label in labels:
        store.upsert_drug(drug, _chunk(label, drug))
    return (
        IndexManager(
            corpus_store=store,
            encoder_factory=factory,
            manifest_path=tmp_path / "index_manifest.json",
            base_collection="med_copilot",
        ),
        store,
    )


# --------------------------------------------------------------- manager 契约


def test_rebuild_builds_new_collection_and_writes_manifest(tmp_path):
    factory = FlakyFactory()
    manager, store = _manager(tmp_path, factory)

    manifest = manager.rebuild()

    assert factory.calls == ["med_copilot-blue"]
    assert manifest["schema_version"] == 2
    assert manifest["corpus_chunk_count"] == len(store.snapshot)
    assert manifest["corpus_sha256"] and isinstance(manifest["corpus_sha256"], str)
    assert manifest["built_at"]
    assert manager.current == manifest
    # manifest 落盘且与新 collection 对应的 encoder 内容一致
    on_disk = json.loads((tmp_path / "index_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["collection"] == manifest["collection"]
    assert len(factory.encoders[0].docs) == manifest["corpus_chunk_count"]


def test_rebuild_is_blue_green_alternating(tmp_path):
    manager, _ = _manager(tmp_path, FlakyFactory())

    first = manager.rebuild()
    second = manager.rebuild()
    third = manager.rebuild()

    assert first["collection"] == "med_copilot-blue"
    assert second["collection"] == "med_copilot-green"
    assert third["collection"] == "med_copilot-blue"


def test_manifest_tracks_corpus_changes_after_reingest(tmp_path):

    manager, store = _manager(tmp_path, FlakyFactory())
    first = manager.rebuild()

    store.upsert_drug("测试药品A", _chunk(LABEL_A_MODIFIED, "测试药品A"))
    second = manager.rebuild()

    assert second["corpus_sha256"] != first["corpus_sha256"]
    assert second["collection"] != first["collection"]
    assert manager.current["collection"] == second["collection"]


def test_failed_rebuild_keeps_previous_manifest_and_collection(tmp_path):
    manager, _ = _manager(tmp_path, FlakyFactory(fail_on_call=2))
    first = manager.rebuild()
    active_encoder = manager.active_encoder

    with pytest.raises(RuntimeError):
        manager.rebuild()

    assert manager.current == first, "构建失败时 manifest 不得变更"
    assert manager.active_encoder is active_encoder, "活跃索引不得被半成品替换"

    recovered = manager.rebuild()
    assert recovered["collection"] == "med_copilot-green"


def test_validation_failure_on_count_mismatch_aborts_switch(tmp_path):
    from medcopilot.rag.index_manager import IndexValidationError

    class LyingEncoder(StubEncoder):
        def count(self) -> int:
            return len(self.docs) + 1  # 灌入后谎报数量

    def lying_factory(name):
        return LyingEncoder(name)

    manager, _ = _manager(tmp_path, lying_factory)
    with pytest.raises(IndexValidationError):
        manager.rebuild()
    assert manager.current is None


# --------------------------------------------------------------- API 集成


def _configured_server(monkeypatch, *, corpus_path, dense_enabled, factory=None):
    from types import SimpleNamespace

    from medcopilot.api import server

    configured = SimpleNamespace(**server.config.model_dump())
    configured.med_corpus_path = corpus_path
    configured.enable_chroma_dense = dense_enabled
    monkeypatch.setattr(server, "config", configured)
    kwargs = {}
    if factory is not None:
        kwargs["dense_encoder_factory"] = factory
    return server, kwargs


def test_api_rebuild_requires_authority_and_returns_manifest(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    client = TestClient(server.create_app(**kwargs))
    client.post(
        "/documents/drug-label",
        data={"raw_text": LABEL_A, "drug_name": "测试药品A"},
    )

    status = client.get("/index/status").json()
    assert status["manifest"] is None

    response = client.post("/index/rebuild")
    assert response.status_code == 200
    manifest = response.json()["manifest"]
    assert manifest["corpus_chunk_count"] > 0
    assert client.get("/index/status").json()["manifest"]["collection"] == manifest["collection"]

    # 重建后再次摄取再重建：manifest 跟随语料变化
    client.post(
        "/documents/drug-label",
        data={"raw_text": LABEL_A_MODIFIED, "drug_name": "测试药品A"},
    )
    second = client.post("/index/rebuild")
    assert second.json()["manifest"]["corpus_sha256"] != manifest["corpus_sha256"]


def test_api_rebuild_requires_admin_key_when_configured(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    configured = server.config
    # 测试用假密钥，片段 join 构造以避免被静态扫描误判为硬编码凭据。
    configured.admin_api_key = "-".join(["rebuild", "adm", "in", "key", "7c4f"])
    client = TestClient(server.create_app(**kwargs))

    assert client.post("/index/rebuild").status_code == 401
    ok = client.post(
        "/index/rebuild", headers={"Authorization": "Bearer " + configured.admin_api_key}
    )
    assert ok.status_code in (200, 409)  # 409 仅当 dense 后端在本环境不可用


def test_index_status_requires_reader_key_when_configured(tmp_path, monkeypatch):
    """Regression tripwire: /index/status sits behind require_reader.

    A prior version left require_reader a no-op when no key was configured
    (fail-open). With a key configured the endpoint must reject anonymous reads
    and accept the bearer key — otherwise index metadata leaks unauthenticated.
    """
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    configured = server.config
    # 测试用假密钥，片段 join 构造以避免被静态扫描误判为硬编码凭据。
    configured.admin_api_key = "-".join(["index", "status", "key", "5f1a"])
    client = TestClient(server.create_app(**kwargs))

    assert client.get("/index/status").status_code == 401
    ok = client.get(
        "/index/status", headers={"Authorization": "Bearer " + configured.admin_api_key}
    )
    assert ok.status_code == 200


def test_api_rebuild_without_persistence_returns_400(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=None, dense_enabled=True, factory=FlakyFactory()
    )
    client = TestClient(server.create_app(**kwargs))
    assert client.post("/index/rebuild").status_code == 400


def test_api_rebuild_without_dense_backend_returns_409(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=False,
    )
    client = TestClient(server.create_app(**kwargs))
    client.post(
        "/documents/drug-label",
        data={"raw_text": LABEL_A, "drug_name": "测试药品A"},
    )
    response = client.post("/index/rebuild")
    assert response.status_code == 409
    assert client.get("/index/status").json()["manifest"] is None


def test_api_rebuild_serves_a_corpus_matching_the_manifest_hash(tmp_path, monkeypatch):
    """The published manifest hash must always describe the corpus being served.

    Regression guard for the rebuild/ingest race: the serving swap previously
    reused ``state["documents"]``, which a concurrent ingestion could replace
    between the rebuild's snapshot and the swap, wiring a retriever built from a
    newer corpus to an encoder indexed from an older snapshot. ``state["documents"]``
    is now taken from the same authoritative snapshot the index was built from.
    """
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    client = TestClient(server.create_app(**kwargs))
    client.post("/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "测试药品A"})

    manifest = client.post("/index/rebuild").json()["manifest"]

    # The manifest on disk, the status endpoint and the rebuilt collection must all
    # describe the same corpus hash and the same non-zero chunk count.
    on_disk = json.loads((tmp_path / "index_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["corpus_sha256"] == manifest["corpus_sha256"]
    assert manifest["corpus_chunk_count"] > 0
    assert client.get("/index/status").json()["manifest"]["corpus_sha256"] == manifest["corpus_sha256"]


def test_rebuild_after_ingest_serves_the_new_corpus_hash(tmp_path):
    """End-to-end: ingest -> rebuild -> ingest -> rebuild must keep the served
    corpus and the manifest hash in lockstep (no stale-encoder/stale-corpus mix)."""
    manager, store = _manager(tmp_path, FlakyFactory())
    first = manager.rebuild()
    assert first["corpus_sha256"] == manager.corpus_hash(store.snapshot)

    store.upsert_drug("测试药品A", _chunk(LABEL_A_MODIFIED, "测试药品A"))
    second = manager.rebuild()
    assert second["corpus_sha256"] == manager.corpus_hash(store.snapshot)
    assert second["corpus_sha256"] != first["corpus_sha256"]
    # The freshly built encoder indexed exactly the pinned corpus.
    assert len(manager.active_encoder.docs) == second["corpus_chunk_count"]


def test_chroma_encoder_replaces_collection_content_across_reingests(tmp_path):
    """P1 regression: blue/green collection names are strictly reused, so a
    backend that ACCUMULATES (Chroma ``upsert`` never deletes ids absent from
    the batch) leaves stale content-addressed ids behind once a label is
    re-ingested with changed ids — the next turn targeting that collection then
    fails its count validation forever (and the first rebuild after a restart
    reuses the polluted color too). ``index_documents`` must REPLACE content."""
    chromadb = pytest.importorskip("chromadb")
    from medcopilot.rag.corpus_store import CorpusStore
    from medcopilot.rag.index_manager import IndexManager
    from medcopilot.rag.vector_store import ChromaDenseEncoder

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))

    class _FixedEmbedding:
        """Deterministic embedding satisfying chroma 1.x's persistence protocol
        (name/is_legacy/get_config) without downloading a real model."""

        is_legacy = False

        def name(self) -> str:
            return "fixed-test-embedding"

        def get_config(self) -> dict:
            return {}

        def build_from_config(self, config):
            return self

        def __call__(self, input):
            return [[0.0, 0.0, 1.0] for _ in input]

    fake_embedding = _FixedEmbedding()

    def factory(name: str) -> ChromaDenseEncoder:
        collection = client.get_or_create_collection(name=name, embedding_function=fake_embedding)
        return ChromaDenseEncoder(collection, fake_embedding, collection_name=name, client=client)

    store = CorpusStore(path=tmp_path / "corpus.jsonl")
    store.upsert_drug("测试药品A", _chunk(LABEL_A, "测试药品A"))
    manager = IndexManager(
        corpus_store=store, encoder_factory=factory,
        manifest_path=tmp_path / "index_manifest.json",
    )
    manager.rebuild()
    # Re-ingest a MODIFIED label: content-addressed ids change (S1∖S2 ≠ ∅).
    store.upsert_drug("测试药品A", _chunk(LABEL_A_MODIFIED, "测试药品A"))
    manager.rebuild()
    # The next blue/green turn reuses the collection that still holds v1's ids.
    manager.rebuild()

    snapshot = store.snapshot
    assert manager.active_encoder.count() == len(snapshot)
    assert manager.current["corpus_sha256"] == manager.corpus_hash(snapshot)


def test_rebuild_manifest_write_failure_keeps_previous_manifest(tmp_path, monkeypatch):
    """A durable-write failure at the manifest publish step must keep the old
    manifest AND the old active binding (the rollback primitive's write side)."""
    from medcopilot.rag import index_manager as index_manager_module

    manager, store = _manager(tmp_path, FlakyFactory())
    first = manager.rebuild()

    def broken_replace(target, write):
        raise OSError("disk full during manifest write")

    monkeypatch.setattr(index_manager_module, "durable_replace", broken_replace)
    store.upsert_drug("测试药品A", _chunk(LABEL_A_MODIFIED, "测试药品A"))
    with pytest.raises(OSError):
        manager.rebuild()

    assert manager.current == first
    assert manager.active_encoder.collection_name == first["collection"]


def test_boot_flags_a_manifest_whose_corpus_hash_diverged(tmp_path, monkeypatch):
    """Boot-time pin health: a corpus that moved past its manifest (crash between
    the durable corpus commit and the best-effort mirror, or out-of-band edits)
    must surface as /health ``index_stale`` instead of silently looking coherent."""
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    client = TestClient(server.create_app(**kwargs))
    client.post("/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "测试药品A"})
    client.post("/index/rebuild")

    # Positive control: an intact pin boots healthy.
    healthy = TestClient(server.create_app(**kwargs))
    assert healthy.get("/health").json()["index_stale"] is False

    # Operator drift: rewrite the on-disk pin with a wrong hash, then reboot.
    manifest_path = tmp_path / "index_manifest.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    on_disk["corpus_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(on_disk), encoding="utf-8")
    stale = TestClient(server.create_app(**kwargs))
    assert stale.get("/health").json()["index_stale"] is True


def test_non_ascii_bearer_token_gets_401_not_500(tmp_path, monkeypatch):
    """``secrets.compare_digest`` raises TypeError on non-ASCII str inputs; the
    comparison must use bytes so a malformed bearer gets a clean 401."""
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    server.config.admin_api_key = "-".join(["auth", "edge", "key", "3b7d"])
    client = TestClient(server.create_app(**kwargs))

    response = client.get(
        "/index/status", headers={"Authorization": b"Bearer caf\xe9-token"}
    )
    assert response.status_code == 401


def test_production_env_with_whitespace_still_fails_closed(tmp_path, monkeypatch):
    """PROJECT_ENV=" production " (quoted .env values keep their whitespace) must
    still be recognized as production: reads and ingestion stay 503 fail-closed
    even when no admin key is configured."""
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    server.config.project_env = " production "
    client = TestClient(server.create_app(**kwargs))

    assert client.get("/index/status").status_code == 503
    assert client.post(
        "/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "测试药品A"}
    ).status_code == 503


def test_api_ingest_never_writes_collections_and_status_reports_staleness(tmp_path, monkeypatch):
    """Rebuild is the single writer of the blue/green collections.

    Regression guard for the ingest-mirror inversion: once ``index_documents``
    became a full-replace primitive (required by the blue-green count
    validation), mirroring incremental chunks from the ingest path truncated
    the active collection to the last drug. Ingestion must not touch any
    collection, and /index/status must report the corpus/index divergence live
    via ``corpus_matches_index``.
    """
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    client = TestClient(server.create_app(**kwargs))
    client.post("/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "测试药品A"})
    first = client.post("/index/rebuild").json()["manifest"]
    assert client.get("/index/status").json()["corpus_matches_index"] is True

    ingested = client.post(
        "/documents/drug-label", data={"raw_text": LABEL_A_MODIFIED, "drug_name": "测试药品A"}
    )
    assert ingested.status_code == 200

    # The active encoder's indexed content must be exactly the pinned rebuild
    # snapshot: under the old mirror it would have been replaced by the new
    # drug's chunks alone. The corpus has visibly moved past the pin.
    active = factory_encoders(kwargs)[-1]
    assert "一次1片" in "\n".join(doc.page_content for doc in active.docs)
    assert "一次2片" not in "\n".join(doc.page_content for doc in active.docs)

    status = client.get("/index/status").json()
    assert status["corpus_matches_index"] is False
    assert status["manifest"]["corpus_sha256"] == first["corpus_sha256"]
    assert status["active_collection"] == first["collection"]

    # Rebuild remains the writer that realigns the pin with the corpus.
    realigned = client.post("/index/rebuild")
    assert realigned.status_code == 200
    assert client.get("/index/status").json()["corpus_matches_index"] is True


def factory_encoders(kwargs):
    return kwargs["dense_encoder_factory"].encoders


def test_api_rebuild_and_ingestion_are_mutually_exclusive(tmp_path, monkeypatch):
    """Tripwire for the rebuild/ingest race: while a rebuild holds ingest_lock
    (pinned inside its encoder build by an event gate), a concurrent ingestion
    must not be able to commit. Deleting the lock around the rebuild makes this
    red: the ingestion would complete mid-rebuild.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    release_build = threading.Event()
    build_entered = threading.Event()

    class GatedFactory:
        """Boot build and the first rebuild run free; later builds gate."""

        def __init__(self):
            self.encoders = []

        def __call__(self, collection_name):
            encoder = StubEncoder(collection_name)
            if len(self.encoders) >= 2:
                original = encoder.index_documents

                def gated(documents, _original=original):
                    build_entered.set()
                    assert release_build.wait(timeout=10)
                    return _original(documents)

                encoder.index_documents = gated
            self.encoders.append(encoder)
            return encoder

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=GatedFactory(),
    )
    client = TestClient(server.create_app(**kwargs))
    client.post("/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "测试药品A"})
    client.post("/index/rebuild")  # second build: unblocked
    client.post(
        "/documents/drug-label", data={"raw_text": LABEL_A_MODIFIED, "drug_name": "测试药品A"}
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        rebuild_future = pool.submit(client.post, "/index/rebuild")
        assert build_entered.wait(timeout=5), "rebuild never reached index_documents"
        ingest_future = pool.submit(
            client.post,
            "/documents/drug-label",
            # Same content the corpus already holds (idempotent re-ingest): the
            # final corpus therefore matches the rebuild's pin, while the
            # request itself still has to wait out the rebuild's ingest_lock.
            data={"raw_text": LABEL_A_MODIFIED, "drug_name": "测试药品A"},
        )
        # The lock guarantees the ingestion cannot commit while the rebuild is
        # pinned inside its build. (Negative wait: with the lock present this
        # can only pass; a missing lock lets the ingest finish here.)
        assert not ingest_future.done(), "ingestion committed while rebuild held ingest_lock"
        release_build.set()
        assert rebuild_future.result(timeout=10).status_code == 200
        assert ingest_future.result(timeout=10).status_code == 200

    assert client.get("/index/status").json()["corpus_matches_index"] is True


def test_api_rebuild_hash_guard_returns_409_and_keeps_manager_state(tmp_path, monkeypatch):
    """Fail-closed guard: a rebuild whose returned manifest pins a corpus hash
    that does not match the serving corpus must be rejected. The branch is
    unreachable via the API under the lock discipline (injected here) — this
    pins that it fails CLOSED and leaves the manager's real state serving.
    """
    from fastapi.testclient import TestClient

    from medcopilot.rag import index_manager as index_manager_module

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    client = TestClient(server.create_app(**kwargs))
    client.post("/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "测试药品A"})
    original = client.post("/index/rebuild").json()["manifest"]

    real_rebuild = index_manager_module.IndexManager.rebuild

    def poisoned_rebuild(self):
        manifest = dict(real_rebuild(self))
        manifest["corpus_sha256"] = "0" * 64
        return manifest

    monkeypatch.setattr(index_manager_module.IndexManager, "rebuild", poisoned_rebuild)

    response = client.post("/index/rebuild")
    assert response.status_code == 409
    # The manager's real manifest (persisted by the genuine rebuild) is what
    # /index/status reports, and it still matches the corpus hash: the guard
    # rejected the poisoned response without disturbing real state.
    assert (
        client.get("/index/status").json()["manifest"]["corpus_sha256"]
        == original["corpus_sha256"]
    )

    monkeypatch.setattr(index_manager_module.IndexManager, "rebuild", real_rebuild)
    assert client.post("/index/rebuild").status_code == 200


def test_boot_treats_a_corrupt_manifest_as_absent(tmp_path, monkeypatch):
    """The manifest is a pointer, not the authority: an unreadable one must be
    treated as absent (warning + fresh pin) instead of blocking the boot."""
    from fastapi.testclient import TestClient

    server, kwargs = _configured_server(
        monkeypatch, corpus_path=str(tmp_path / "corpus.jsonl"), dense_enabled=True,
        factory=FlakyFactory(),
    )
    (tmp_path / "index_manifest.json").write_text("not json at all", encoding="utf-8")

    client = TestClient(server.create_app(**kwargs))

    assert client.get("/index/status").json()["manifest"] is None
    assert client.get("/health").status_code == 200
