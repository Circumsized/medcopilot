"""权威语料外置与幂等摄取的契约测试（Slice 2）。

这些测试先于实现存在：语料的权威副本必须落在进程外（JSONL），摄取必须
按药品原子替换且幂等；写盘失败不得污染既有权威文件或内存快照。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SAMPLE_LABEL_A = (
    "【药品名称】测试药品A\n"
    "【适应症】用于缓解轻度疼痛。\n"
    "【用法用量】口服。成人一次1片，一日3次。\n"
    "【禁忌】对本品过敏者禁用。\n"
)
SAMPLE_LABEL_A_MODIFIED = (
    "【药品名称】测试药品A\n"
    "【适应症】用于缓解中度疼痛。\n"
    "【用法用量】口服。成人一次2片，一日3次。\n"
)
SAMPLE_LABEL_B = (
    "【药品名称】测试药品B\n"
    "【禁忌】儿童禁用。\n"
)


def _chunk(label: str, drug: str):
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    return ClinicalChunker.chunk_drug_label(label, drug_name=drug)


def _chunk_ids(docs) -> set[str]:
    return {d.metadata["chunk_id"] for d in docs}


def _drug_names(docs) -> set[str]:
    return {d.metadata["drug_name"] for d in docs}


# --------------------------------------------------------------- store 契约


def test_store_round_trip_preserves_chunks_across_instances(tmp_path):
    from medcopilot.rag.corpus_store import CorpusStore

    path = tmp_path / "corpus.jsonl"
    store = CorpusStore(path=path)
    docs = _chunk(SAMPLE_LABEL_A, "测试药品A")
    store.upsert_drug("测试药品A", docs)

    reloaded = CorpusStore(path=path)
    loaded = reloaded.load()

    assert _chunk_ids(loaded) == _chunk_ids(docs)
    assert all(isinstance(d.page_content, str) and d.page_content for d in loaded)
    assert _drug_names(loaded) == {"测试药品A"}


def test_reingesting_identical_label_is_idempotent(tmp_path):
    from medcopilot.rag.corpus_store import CorpusStore

    store = CorpusStore(path=tmp_path / "corpus.jsonl")
    docs = _chunk(SAMPLE_LABEL_A, "测试药品A")
    store.upsert_drug("测试药品A", docs)
    before = list(store.snapshot)

    added, replaced = store.upsert_drug("测试药品A", _chunk(SAMPLE_LABEL_A, "测试药品A"))

    assert added == 0
    assert replaced == 0
    assert store.snapshot == before
    # 落盘文件也不应增长
    lines = (tmp_path / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(before)


def test_reingesting_modified_label_replaces_only_that_drug(tmp_path):
    from medcopilot.rag.corpus_store import CorpusStore

    store = CorpusStore(path=tmp_path / "corpus.jsonl")
    store.upsert_drug("测试药品A", _chunk(SAMPLE_LABEL_A, "测试药品A"))
    store.upsert_drug("测试药品B", _chunk(SAMPLE_LABEL_B, "测试药品B"))
    old_by_section = {
        d.metadata["section"]: d.metadata["chunk_id"]
        for d in store.snapshot
        if d.metadata["drug_name"] == "测试药品A"
    }

    store.upsert_drug("测试药品A", _chunk(SAMPLE_LABEL_A_MODIFIED, "测试药品A"))

    new_by_section = {
        d.metadata["section"]: d.metadata["chunk_id"]
        for d in store.snapshot
        if d.metadata["drug_name"] == "测试药品A"
    }
    # 已删除章节：旧 chunk 必须消失，不得残留
    assert "禁忌" not in new_by_section
    # 内容变化的章节：chunk 更新为新 ID
    assert new_by_section["适应症"] != old_by_section["适应症"]
    assert new_by_section["用法用量"] != old_by_section["用法用量"]
    # 内容未变的章节：幂等保留同一 chunk
    assert new_by_section["药品名称"] == old_by_section["药品名称"]
    # 其他药品不受影响
    assert _drug_names(store.snapshot) == {"测试药品A", "测试药品B"}
    b_chunks = [d for d in store.snapshot if d.metadata["drug_name"] == "测试药品B"]
    assert b_chunks, "其他药品不得受影响"


def test_chunk_ids_are_stable_across_processes(tmp_path):
    docs_first = _chunk(SAMPLE_LABEL_A, "测试药品A")
    docs_second = _chunk(SAMPLE_LABEL_A, "测试药品A")
    assert _chunk_ids(docs_first) == _chunk_ids(docs_second)


def test_failed_persist_keeps_previous_file_and_snapshot(tmp_path, monkeypatch):
    from medcopilot.rag.corpus_store import CorpusStore
    path = tmp_path / "corpus.jsonl"
    store = CorpusStore(path=path)
    store.upsert_drug("测试药品A", _chunk(SAMPLE_LABEL_A, "测试药品A"))
    good_snapshot = list(store.snapshot)
    good_bytes = path.read_bytes()

    def broken_replace(target, write):
        raise OSError("disk full")

    # Persistence now goes through the shared durable-replace helper; failing the
    # publish step must leave the authoritative file and the in-memory snapshot
    # untouched (write-through semantics). Patch it on the consuming module, since
    # corpus_store.py binds the name at import time.
    monkeypatch.setattr("medcopilot.rag.corpus_store.durable_replace", broken_replace)
    with pytest.raises(OSError):
        store.upsert_drug("测试药品B", _chunk(SAMPLE_LABEL_B, "测试药品B"))

    assert path.read_bytes() == good_bytes, "权威文件不得被半写或截断"
    assert store.snapshot == good_snapshot, "写盘失败时内存快照不得变更"

    # 磁盘恢复后同一实例可继续工作
    monkeypatch.undo()
    store.upsert_drug("测试药品B", _chunk(SAMPLE_LABEL_B, "测试药品B"))
    assert "测试药品B" in _drug_names(store.snapshot)


def test_corrupt_lines_are_skipped_and_reported(tmp_path):
    from medcopilot.rag.corpus_store import CorpusStore

    path = tmp_path / "corpus.jsonl"
    store = CorpusStore(path=path)
    store.upsert_drug("测试药品A", _chunk(SAMPLE_LABEL_A, "测试药品A"))
    good_line = path.read_text(encoding="utf-8").splitlines()[0]

    path.write_text(
        "not-json-at-all\n"
        + json.dumps({"page_content": 123, "metadata": {}}) + "\n"
        + json.dumps({"page_content": "缺药名的记录", "metadata": {}}) + "\n"
        + good_line + "\n",
        encoding="utf-8",
    )

    reloaded = CorpusStore(path=path)
    loaded = reloaded.load()
    assert reloaded.degraded is True
    assert len(loaded) == 1
    assert loaded[0].metadata["drug_name"] == "测试药品A"


def test_duplicate_chunk_id_keeps_later_record_and_flags_degraded(tmp_path):
    from medcopilot.rag.corpus_store import CorpusStore

    path = tmp_path / "corpus.jsonl"
    duplicated_id = "dup-chunk-0001"
    path.write_text(
        json.dumps({
            "page_content": "【药品名称】测试药品A\n【用法用量】口服。成人一次1片，一日3次。",
            "metadata": {"chunk_id": duplicated_id, "drug_name": "测试药品A"},
        }, ensure_ascii=False) + "\n"
        + json.dumps({
            "page_content": "【药品名称】测试药品A\n【用法用量】口服。成人一次2片，一日3次。",
            "metadata": {"chunk_id": duplicated_id, "drug_name": "测试药品A"},
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    store = CorpusStore(path=path)
    loaded = store.load()

    # 后写覆盖语义不变：重复 chunk_id 只保留后一条记录
    assert len(store.snapshot) == 1
    assert len(loaded) == 1
    assert store.snapshot[0].page_content.endswith("成人一次2片，一日3次。")
    # 但重复导致一条记录被覆盖丢失，必须在 degraded 标志上可见，
    # /health 不得在此情况下谎报语料健康
    assert store.degraded is True


def test_store_without_path_stays_pure_in_memory():
    from medcopilot.rag.corpus_store import CorpusStore

    store = CorpusStore(path=None)
    store.upsert_drug("测试药品A", _chunk(SAMPLE_LABEL_A, "测试药品A"))
    assert store.persisted is False
    assert len(store.snapshot) > 0


# --------------------------------------------------------------- API 集成


def _configured_server(monkeypatch, corpus_path: str | None):
    from types import SimpleNamespace

    from medcopilot.api import server

    configured = SimpleNamespace(**server.config.model_dump())
    configured.med_corpus_path = corpus_path
    monkeypatch.setattr(server, "config", configured)
    return server


def test_api_ingestion_persists_and_survives_app_restart(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _configured_server(monkeypatch, str(tmp_path / "corpus.jsonl"))
    corpus_path = Path(server.config.med_corpus_path)

    first = TestClient(server.create_app())
    response = first.post(
        "/documents/drug-label",
        data={"raw_text": SAMPLE_LABEL_A, "drug_name": "测试药品A"},
    )
    assert response.status_code == 200
    assert response.json()["corpus_size"] == response.json()["chunks_added"] > 0
    assert corpus_path.exists(), "摄取必须落盘到权威语料文件"

    second = TestClient(server.create_app())
    health = second.get("/health").json()
    assert health["corpus_documents"] == response.json()["corpus_size"]

    consult = second.post("/consultations", json={"query": "测试药品A的用法用量是什么？"})
    assert consult.status_code == 200


def test_api_double_ingestion_does_not_grow_corpus(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _configured_server(monkeypatch, str(tmp_path / "corpus.jsonl"))
    client = TestClient(server.create_app())

    first = client.post(
        "/documents/drug-label",
        data={"raw_text": SAMPLE_LABEL_A, "drug_name": "测试药品A"},
    )
    size_after_first = first.json()["corpus_size"]
    second = client.post(
        "/documents/drug-label",
        data={"raw_text": SAMPLE_LABEL_A, "drug_name": "测试药品A"},
    )
    assert second.status_code == 200
    assert second.json()["chunks_added"] == 0
    assert second.json()["corpus_size"] == size_after_first


def test_health_reports_corpus_state(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _configured_server(monkeypatch, str(tmp_path / "corpus.jsonl"))
    client = TestClient(server.create_app())
    health = client.get("/health").json()
    assert health["corpus_persisted"] is True
    assert health["corpus_degraded"] is False


def test_health_reports_degraded_after_corrupt_boot(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_bytes(b"garbage-line\n\xff\xfeinvalid-utf8\n")
    server = _configured_server(monkeypatch, str(corpus_path))

    client = TestClient(server.create_app())
    health = client.get("/health").json()
    assert health["corpus_degraded"] is True
    assert health["corpus_documents"] == 0


def test_corpus_store_loads_utf8_bom_first_line_without_data_loss(tmp_path):
    from medcopilot.rag.corpus_store import CorpusStore

    corpus_path = tmp_path / "bom_corpus.jsonl"
    store = CorpusStore(corpus_path)
    chunks = _chunk(SAMPLE_LABEL_A, "测试药品A")
    store.upsert_drug("测试药品A", chunks)
    raw = corpus_path.read_bytes()
    corpus_path.write_bytes(b"\xef\xbb\xbf" + raw)

    reloaded = CorpusStore(corpus_path)
    loaded_docs = reloaded.load()
    assert reloaded.degraded is False
    assert len(loaded_docs) == len(chunks)


def test_persistence_disabled_keeps_slice1_behavior(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _configured_server(monkeypatch, None)
    client = TestClient(server.create_app())
    response = client.post(
        "/documents/drug-label",
        data={"raw_text": SAMPLE_LABEL_A, "drug_name": "测试药品A"},
    )
    assert response.status_code == 200
    assert not (tmp_path / "corpus.jsonl").exists()

    health = client.get("/health").json()
    assert health["corpus_persisted"] is False


def test_injected_initial_documents_bypass_disk(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    server = _configured_server(monkeypatch, str(tmp_path / "corpus.jsonl"))
    injected = _chunk(SAMPLE_LABEL_B, "测试药品B")
    client = TestClient(server.create_app(initial_documents=injected))

    assert client.get("/health").json()["corpus_documents"] == len(injected)
    assert not (tmp_path / "corpus.jsonl").exists(), "显式注入不得触发磁盘读写"
