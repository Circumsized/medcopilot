"""单机恢复演练（Slice 8 + W5.5）。

把 docs/RELEASE_CHECKLIST.md 的恢复演练表变成可重复执行的自动化：八类
已知故障场景端到端执行并断言期望行为，出演练报告。全部使用临时目录与
进程内应用实例，不触碰真实数据、不改变任何门控行为。

场景：
1. corpus_corrupt_lines_degrade_not_fail —— 权威语料坏行：服务降级可用。
2. index_build_failure_keeps_previous_manifest —— 索引构建失败：旧状态保持。
3. unknown_session_404_but_emergency_issued —— 未知会话 404；急救签发新标识。
4. broken_policy_artifact_failsafe —— 策略工件损坏：健康如实报告且门控不变。
5. unrecognized_env_fails_closed —— PROJECT_ENV 未识别：读写 fail-closed。
6. audit_chain_tamper_detected —— 审计哈希链可重放校验，篡改历史必被检出。
7. backup_restore_recovers_serving_state —— 备份→数据丢失→恢复→服务一致
   (W5.5；detail.rto_seconds 是 RTO 的实测证据)。
8. backup_tamper_fails_closed —— 快照被篡改：恢复拒绝且目标文件零触碰。
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# W1.3 default-deny: the drill exercises the local dev flow; an empty
# PROJECT_ENV would fail closed (503) before the emergency short-circuit.
# Must precede any project import that transitively instantiates AppConfig().
# Also isolate AUDIT_LOG_PATH and LOG_FILE into a process-scoped temp dir so
# running the drill as a standalone script never appends synthetic events to
# the repository's real logs/audit.jsonl chain.
os.environ.setdefault("PROJECT_ENV", "development")
os.environ.setdefault("RERANK_ENABLED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
_DRILL_ENV_TMP = Path(tempfile.mkdtemp(prefix=f"recovery-drill-env-{os.getpid()}-"))
atexit.register(shutil.rmtree, _DRILL_ENV_TMP, ignore_errors=True)
os.environ.setdefault("AUDIT_LOG_PATH", str(_DRILL_ENV_TMP / "audit.jsonl"))
os.environ.setdefault("LOG_FILE", str(_DRILL_ENV_TMP / "app.log"))

from medcopilot.common.fsutil import atomic_write_json  # noqa: E402

REPORT_SCHEMA_VERSION = 1

LABEL = "【药品名称】演练药品\n【用法用量】口服。一次1片，一日3次。\n【禁忌】过敏者禁用。\n"
EMERGENCY_QUERY = "突发心肌梗死，胸痛压榨濒死感"
PLAIN_QUERY = "演练药品的用法用量是什么？"


def _configure(**overrides):
    """替换 server.config 为覆盖副本，返回还原函数。"""
    from medcopilot.api import server

    original = server.config
    configured = SimpleNamespace(**original.model_dump())
    if "audit_log_path" not in overrides:
        configured.audit_log_path = str(_DRILL_ENV_TMP / f"audit-{time.monotonic_ns()}.jsonl")
    for key, value in overrides.items():
        setattr(configured, key, value)
    server.config = configured

    def restore() -> None:
        server.config = original

    return restore


def _scenario_corpus_corrupt(tmp: Path) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from medcopilot.api import server

    corpus = tmp / "corpus.jsonl"
    corpus.write_text("not-json-at-all\n", encoding="utf-8")
    restore = _configure(med_corpus_path=str(corpus))
    try:
        client = TestClient(server.create_app())
        # 预置一条好语料：直接走摄取（写盘会把好行与坏行共存吗？摄取重写整个文件，
        # 所以改为摄取后手工注入坏行，再重建 app 模拟崩溃后重启。）
        ingest = client.post(
            "/documents/drug-label", data={"raw_text": LABEL, "drug_name": "演练药品"}
        )
        if ingest.status_code != 200:
            # 显式短路而非 assert：python -O 会剥离 assert，演练会在预置失败的
            # 基础上空转成假通过。
            return {
                "scenario": "corpus_corrupt_lines_degrade_not_fail",
                "passed": False,
                "detail": {"ingest_status": ingest.status_code},
            }
        with corpus.open("a", encoding="utf-8") as handle:
            handle.write("corrupted-line-not-json\n")
        restarted = TestClient(server.create_app())
        health = restarted.get("/health").json()
        consult = restarted.post("/consultations", json={"query": PLAIN_QUERY})
        return {
            "scenario": "corpus_corrupt_lines_degrade_not_fail",
            "passed": bool(
                health["corpus_degraded"] is True
                and health["corpus_documents"] >= 1
                and consult.status_code == 200
                and consult.json()["session_id"]
            ),
            "detail": {
                "corpus_degraded": health["corpus_degraded"],
                "corpus_documents": health["corpus_documents"],
                "consult_status": consult.status_code,
            },
        }
    finally:
        restore()


class _FailAfterStableBuildFactory:
    """前两次构建成功（boot + 首次 rebuild 真实建立 manifest），之后每次构建注入失败。

    场景 2 必须演练**真回滚**：先有一个成功 rebuild 写入的 manifest 与活跃
    encoder，再让下一次构建失败，才能证明"旧 manifest/encoder 原样维持"。
    只在 boot 之后失败的工厂从未建立过 manifest，那种 500 证明不了回滚。
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, name: str):
        self.calls += 1
        if self.calls <= 2:
            return _StubEncoder(name)
        raise RuntimeError("drill-injected encoder build failure")


class _StubEncoder:
    """encoder 协议最小实现（无 chromadb 依赖）。

    ``indexed_ids`` 不是可选装饰：manifest 的 ``index_content_sha256`` 与
    ``bind_active`` 的 live-id 校验都经它证明"集合里装的正是这批 chunk"，
    没有它演练只走到 count 比对，证不住内容身份。
    """

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        self.docs: list = []

    def index_documents(self, documents) -> int:
        self.docs = list(documents)
        return len(self.docs)

    def count(self) -> int:
        return len(self.docs)

    def indexed_ids(self) -> list[str]:
        return [
            str(doc.metadata.get("chunk_id") or f"doc-{index}")
            for index, doc in enumerate(self.docs)
        ]


def _scenario_index_failure(tmp: Path) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from medcopilot.api import server

    corpus = tmp / "corpus.jsonl"
    restore = _configure(
        med_corpus_path=str(corpus), enable_chroma_dense=True,
        # 本场景连发两次 rebuild(先成功建 manifest、再注入失败)。限流是运营可调的
        # 滥用控制(默认 2 次/小时),不是本场景的演练对象;关掉它只是避免"配额调成 1"
        # 时把回滚演练误判成 429 假红(与 conftest 关闭限流同款理由)。
        rate_limit_enabled=False,
    )
    try:
        client = TestClient(server.create_app(dense_encoder_factory=_FailAfterStableBuildFactory()))
        ingest = client.post(
            "/documents/drug-label", data={"raw_text": LABEL, "drug_name": "演练药品"}
        )
        if ingest.status_code != 200:
            return {
                "scenario": "index_build_failure_keeps_previous_manifest",
                "passed": False,
                "detail": {"ingest_status": ingest.status_code},
            }
        # 先真实成功一次：manifest 与活跃 encoder 由此建立（蓝绿第一色）。
        first = client.post("/index/rebuild")
        if first.status_code != 200:
            return {
                "scenario": "index_build_failure_keeps_previous_manifest",
                "passed": False,
                "detail": {"first_rebuild_status": first.status_code},
            }
        first_manifest = first.json()["manifest"]
        # 再注入构建失败：旧 manifest/encoder 必须原样维持，consult 仍可用。
        rebuild = client.post("/index/rebuild")
        status = client.get("/index/status").json()
        consult = client.post("/consultations", json={"query": PLAIN_QUERY})
        manifest_unchanged = status["manifest"] == first_manifest
        return {
            "scenario": "index_build_failure_keeps_previous_manifest",
            "passed": bool(
                rebuild.status_code == 500
                and manifest_unchanged
                and status["active_collection"] == first_manifest["collection"]
                # stub 没有嵌入身份：manifest 如实落 unverified，演练不假冒身份。
                and status["manifest_status"] == "unverified"
                # digest 路径真实走过（StubEncoder.indexed_ids 的证明）。
                and bool(first_manifest.get("index_content_sha256"))
                and consult.status_code == 200
                and consult.json()["session_id"]
            ),
            "detail": {
                "first_rebuild_status": first.status_code,
                "rebuild_status": rebuild.status_code,
                "manifest_unchanged": manifest_unchanged,
                "active_collection": status["active_collection"],
                "manifest_status": status["manifest_status"],
                "digest_recorded": bool(first_manifest.get("index_content_sha256")),
                "consult_status": consult.status_code,
            },
        }
    finally:
        restore()


def _scenario_sessions(tmp: Path) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from medcopilot.api import server

    restore = _configure(med_corpus_path="")
    try:
        client = TestClient(server.create_app())
        unknown = client.post(
            "/consultations", json={"query": PLAIN_QUERY, "session_id": "drill-unknown"}
        )
        emergency = client.post(
            "/consultations", json={"query": EMERGENCY_QUERY, "session_id": "drill-unknown"}
        )
        body = emergency.json()
        return {
            "scenario": "unknown_session_404_but_emergency_issued",
            "passed": bool(
                unknown.status_code == 404
                and emergency.status_code == 200
                and body["is_emergency"] is True
                and body["session_id"]
                and body["session_id"] != "drill-unknown"
            ),
            "detail": {
                "unknown_status": unknown.status_code,
                "emergency_status": emergency.status_code,
                "fresh_session_issued": bool(body.get("session_id")),
            },
        }
    finally:
        restore()


def _scenario_policy_failsafe(tmp: Path) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from medcopilot.api import server

    broken = tmp / "broken_policy.json"
    broken.write_text("{not json", encoding="utf-8")
    restore = _configure(safety_policy_path=str(broken))
    try:
        client = TestClient(server.create_app())
        health = client.get("/health").json()
        emergency = client.post("/consultations", json={"query": EMERGENCY_QUERY})
        return {
            "scenario": "broken_policy_artifact_failsafe",
            "passed": bool(
                health["policy_review_status"] == "invalid"
                and health["policy_version"] is None
                and emergency.status_code == 200
                and emergency.json()["is_emergency"] is True
            ),
            "detail": {
                "policy_review_status": health["policy_review_status"],
                "emergency_status": emergency.status_code,
                "emergency_short_circuit": emergency.json()["is_emergency"],
            },
        }
    finally:
        restore()


def _scenario_env_unrecognized_fails_closed(tmp: Path) -> dict[str, Any]:
    """W1.3:漏配/错拼 PROJECT_ENV ⇒ 读写 503 fail-closed,即便密钥已配置。"""
    from fastapi.testclient import TestClient

    from medcopilot.api import server

    restore = _configure(project_env="prodction", admin_api_key="drill-key")
    try:
        client = TestClient(server.create_app())
        health = client.get("/health").json()
        read = client.post(
            "/consultations",
            json={"query": PLAIN_QUERY},
            headers={"Authorization": "Bearer drill-key"},
        )
        ingest = client.post(
            "/documents/drug-label",
            data={"raw_text": LABEL, "drug_name": "演练药品"},
            headers={"Authorization": "Bearer drill-key"},
        )
        return {
            "scenario": "unrecognized_env_fails_closed",
            "passed": bool(
                health["env_recognized"] is False
                and read.status_code == 503
                and ingest.status_code == 503
            ),
            "detail": {
                "env_class": health["env_class"],
                "read_status": read.status_code,
                "ingest_status": ingest.status_code,
            },
        }
    finally:
        restore()


def _scenario_audit_chain_replay(tmp: Path) -> dict[str, Any]:
    """W1.2:审计哈希链可重放校验;篡改历史记录必须被检出。"""
    from medcopilot.audit import AuditLogger

    sink = tmp / "audit.jsonl"
    audit = AuditLogger(str(sink))
    for index in range(3):
        audit.record(
            event_type="consultation", endpoint="/consultations", session_hash="anonymous",
            key_id="drill-key-id", intent="medical", is_emergency=False, abstained=False,
            evidence_chunk_ids=(), evidence_count=0, policy_version=2,
            latency_ms=float(index), status_code=200,
        )
    ok, _ = audit.verify_chain()
    tampered_ok = ok
    if ok:
        lines = sink.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[1])
        record["abstained"] = True
        lines[1] = json.dumps(record, ensure_ascii=False)
        sink.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tampered_ok, broken_line = AuditLogger(str(sink)).verify_chain()
        detected = (not tampered_ok) and broken_line == 2
    else:
        detected = False
    return {
        "scenario": "audit_chain_tamper_detected",
        "passed": bool(ok and detected),
        "detail": {"intact_chain": ok, "tamper_detected": detected},
    }


def _scenario_backup_restore(tmp: Path) -> dict[str, Any]:
    """W5.5 场景 7:备份→注入数据丢失→从快照恢复→重建 app→语料与答案一致。

    detail.rto_seconds 是 RTO 的实测证据(随 recovery_drill.json 进 release
    evidence)。恢复用 backup_snapshot.restore_snapshot 直调(sha256 校验 +
    临时文件 + os.replace 原子发布),演练本身绝不用 assert 判定。
    """
    from fastapi.testclient import TestClient

    from medcopilot.api import server
    from medcopilot.scripts import backup_snapshot

    corpus = tmp / "corpus.jsonl"
    restore = _configure(
        med_corpus_path=str(corpus), audit_log_path=str(tmp / "audit.jsonl")
    )
    try:
        client = TestClient(server.create_app())
        ingest = client.post(
            "/documents/drug-label", data={"raw_text": LABEL, "drug_name": "演练药品"}
        )
        if ingest.status_code != 200:
            return {
                "scenario": "backup_restore_recovers_serving_state",
                "passed": False,
                "detail": {"ingest_status": ingest.status_code},
            }
        before = client.post("/consultations", json={"query": PLAIN_QUERY}).json()

        started = time.perf_counter()
        snapshot_result = backup_snapshot.take_snapshot(corpus, None, tmp / "backups")
        # 注入数据丢失:权威语料被整文件删除(模拟误删/磁盘故障)。
        corpus.unlink()
        restore_result = backup_snapshot.restore_snapshot(
            Path(snapshot_result["snapshot_dir"]), corpus
        )
        rebuilt = TestClient(server.create_app())
        health = rebuilt.get("/health").json()
        after = rebuilt.post("/consultations", json={"query": PLAIN_QUERY}).json()
        rto_seconds = time.perf_counter() - started

        return {
            "scenario": "backup_restore_recovers_serving_state",
            "passed": bool(
                restore_result["verified"] is True
                and restore_result["restored"] == str(corpus)
                and health["corpus_degraded"] is False
                and health["corpus_documents"] >= 1
                and after["answer"] == before["answer"]
                and after["session_id"]
            ),
            "detail": {
                "rto_seconds": round(rto_seconds, 2),
                "restored_files": restore_result["verified_files"],
                "snapshot_files": len(snapshot_result["files"]),
                "corpus_documents": health["corpus_documents"],
                "corpus_degraded": health["corpus_degraded"],
                "answer_identical": after["answer"] == before["answer"],
            },
        }
    finally:
        restore()


def _scenario_backup_tamper(tmp: Path) -> dict[str, Any]:
    """W5.5 场景 8:快照被篡改 ⇒ 恢复 fail-closed,目标文件零触碰。

    断言目标文件内容与 mtime 都不变——恢复损坏数据比没有恢复更糟。
    """
    from medcopilot.scripts import backup_snapshot

    corpus = tmp / "tamper_corpus.jsonl"
    corpus.write_text('{"good": "line"}\n', encoding="utf-8")
    snapshot_result = backup_snapshot.take_snapshot(corpus, None, tmp / "backups")
    snapshot_dir = Path(snapshot_result["snapshot_dir"])

    # 篡改快照内的语料副本(manifest 里的 sha256 不再匹配)。
    (snapshot_dir / "corpus.jsonl").write_text('{"tampered": "line"}\n', encoding="utf-8")
    before_bytes = corpus.read_bytes()
    before_mtime = corpus.stat().st_mtime_ns

    verify_ok, verify_errors = backup_snapshot.verify(snapshot_dir)
    result = backup_snapshot.restore_snapshot(snapshot_dir, corpus)
    time.sleep(0.01)
    untouched = (
        corpus.read_bytes() == before_bytes
        and corpus.stat().st_mtime_ns == before_mtime
        and result["restored"] is None
    )
    return {
        "scenario": "backup_tamper_fails_closed",
        "passed": bool(
            verify_ok is False
            and any(error.startswith("hash:") for error in verify_errors)
            and result["verified"] is False
            and untouched
        ),
        "detail": {
            "verify_passed": verify_ok,
            "hash_errors": [e for e in verify_errors if e.startswith("hash:")],
            "restore_verified": result["verified"],
            "restore_restored": result["restored"],
            "target_untouched": untouched,
        },
    }


SCENARIO_SPECS = (
    ("corpus_corrupt_lines_degrade_not_fail", _scenario_corpus_corrupt),
    ("index_build_failure_keeps_previous_manifest", _scenario_index_failure),
    ("unknown_session_404_but_emergency_issued", _scenario_sessions),
    ("broken_policy_artifact_failsafe", _scenario_policy_failsafe),
    ("unrecognized_env_fails_closed", _scenario_env_unrecognized_fails_closed),
    ("audit_chain_tamper_detected", _scenario_audit_chain_replay),
    ("backup_restore_recovers_serving_state", _scenario_backup_restore),
    ("backup_tamper_fails_closed", _scenario_backup_tamper),
)
SCENARIOS = tuple(fn for _, fn in SCENARIO_SPECS)


def run_drill() -> dict[str, Any]:
    results = []
    with tempfile.TemporaryDirectory(prefix="recovery-drill-") as tmp_name:
        tmp = Path(tmp_name)
        for scenario_id, scenario in SCENARIO_SPECS:
            started = time.perf_counter()
            try:
                result = scenario(tmp)
            except Exception as exc:
                result = {
                    "scenario": scenario_id,
                    "passed": False,
                    "detail": {"error_type": type(exc).__name__, "error": str(exc)},
                }
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            results.append(result)
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "scenarios": results,
        "all_passed": all(item["passed"] for item in results),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="single-machine recovery drill")
    parser.add_argument("--output", default="outputs/recovery_drill.json")
    args = parser.parse_args(argv)

    report = run_drill()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)

    for item in report["scenarios"]:
        mark = "PASS" if item["passed"] else "FAIL"
        print(f"  [{mark}] {item['scenario']} ({item['elapsed_seconds']}s)")
    print(f"RECOVERY DRILL: all_passed={report['all_passed']} report={out}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
