"""W1.2 审计日志契约:哈希链防篡改、免 PHI、fail-safe、端点集成。

设计依据:HIPAA §164.312(b)(记录+审查)、45 CFR §164.316(b)(2)(i)(6 年保留)、
ONC/ASTM E2147-18(法医重建粒度、NTP 时钟)、FDA Part 11(防篡改哈希链)。
"""

from __future__ import annotations

import json

import pytest

from medcopilot.audit import AuditLogger, derive_key_id, hash_session


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


def _fields(**overrides):
    base = {
        "event_type": "consultation",
        "endpoint": "/consultations",
        "session_hash": "anonymous",
        "key_id": "k" * 16,
        "intent": "medical",
        "is_emergency": False,
        "abstained": False,
        "evidence_chunk_ids": (),
        "evidence_count": 0,
        "policy_version": 2,
        "latency_ms": 1.0,
        "status_code": 200,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------ 单元:链与字段


def test_hash_chain_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLogger(str(path))
    for _ in range(3):
        audit.record(**_fields())
    ok, broken_line = audit.verify_chain()
    assert ok is True and broken_line == 0

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["abstained"] = True  # 篡改第 2 条
    lines[1] = json.dumps(record, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, broken_line = AuditLogger(str(path)).verify_chain()
    assert ok is False and broken_line == 2


def test_chain_head_survives_restart(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditLogger(str(path))
    first.record(**_fields())
    head = first.chain_head()
    reopened = AuditLogger(str(path))
    assert reopened.chain_head() == head
    reopened.record(**_fields())
    assert reopened.chain_head() != head


def test_records_are_phi_free(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = AuditLogger(str(path))
    audit.record(**_fields(evidence_chunk_ids=("chunk-abc123",)))
    line = path.read_text(encoding="utf-8")
    # 只允许出现结构字段;毒句 query/answer 文本绝不出现在审计里
    assert "喷射性呕吐" not in line
    assert "布洛芬" not in line
    record = json.loads(line)
    assert record["evidence_chunk_ids"] == ["chunk-abc123"]


def test_key_derivation_is_stable_and_not_the_key():
    key_id = derive_key_id("super-secret-token")
    assert key_id == derive_key_id("super-secret-token")
    assert key_id != derive_key_id("other-token")
    assert "super-secret-token" not in key_id
    assert derive_key_id("") == "anonymous"


def test_session_hash_is_one_way():
    digest = hash_session("session-uuid-1234")
    assert digest != "session-uuid-1234"
    assert len(digest) == 16


def test_unwritable_sink_counts_failures(tmp_path):
    # 路径指向一个目录 ⇒ 打开必然失败;record 不抛错,failures 递增,ready=False
    audit = AuditLogger(str(tmp_path))
    assert audit.record(**_fields()) is None
    assert audit.failures == 1
    assert audit.ready is False


# ------------------------------------------------------------ 集成:API 埋点


def _app_with_audit(monkeypatch, tmp_path, **override_env):
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from medcopilot.api import server

    configured = SimpleNamespace(**server.config.model_dump())
    configured.audit_log_path = str(tmp_path / "audit.jsonl")
    configured.project_env = "development"
    for key, value in override_env.items():
        setattr(configured, key, value)
    monkeypatch.setattr(server, "config", configured)
    return TestClient(server.create_app()), tmp_path / "audit.jsonl"


def test_consultation_and_emergency_are_audited(monkeypatch, tmp_path):
    client, audit_path = _app_with_audit(monkeypatch, tmp_path)

    r = client.post("/consultations", json={"query": "布洛芬缓释胶囊的用法用量是什么"})
    assert r.status_code == 200
    r = client.post("/consultations", json={"query": "家人突然心梗发作"})
    assert r.status_code == 200
    assert r.json()["is_emergency"] is True

    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    types = [e["event_type"] for e in events]
    assert "consultation" in types and "emergency_triage" in types
    emergency = next(e for e in events if e["event_type"] == "emergency_triage")
    assert "cv.mi" in emergency["red_rule_ids"]
    assert emergency["is_emergency"] is True
    consultation = next(e for e in events if e["event_type"] == "consultation")
    assert consultation["policy_version"] >= 1
    assert consultation["key_id"] != ""
    # 免 PHI:审计文件不含 query 原文
    raw = audit_path.read_text(encoding="utf-8")
    assert "布洛芬缓释胶囊" not in raw and "心梗" not in raw


def test_health_exposes_audit_surface(monkeypatch, tmp_path):
    client, _ = _app_with_audit(monkeypatch, tmp_path)
    health = client.get("/health").json()
    for key in (
        "audit_ready", "audit_sink", "audit_records_today", "audit_failures",
        "audit_chain_head", "audit_retention_days", "audit_time_offset_ms",
    ):
        assert key in health, f"/health missing {key}"
    assert health["audit_ready"] is True
    assert health["audit_retention_days"] == 2190


def test_ingest_is_audited(monkeypatch, tmp_path):
    client, audit_path = _app_with_audit(monkeypatch, tmp_path)
    r = client.post(
        "/documents/drug-label",
        data={"raw_text": "【药品名称】审计测试片\n【适应症】测试。", "drug_name": "审计测试片"},
    )
    assert r.status_code in (200, 201)
    events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    ingest = [e for e in events if e["event_type"] == "ingest"]
    assert ingest and ingest[-1]["evidence_count"] >= 1
