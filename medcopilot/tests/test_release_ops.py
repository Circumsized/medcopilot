"""Slice 7 契约测试：发布证据包、负载检查与分位数统计。

证据包只聚合不修饰：必须门失败即整体失败；eval_rag 的诚实 FAIL 按
``expected_fail`` 记录且不改变整体结论。负载检查只做连接层健康断言。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# release_pack 是工作区级聚合门禁（仓库根 scripts/）；load_check 属本项目。
SCRIPT_PATHS = {
    "release_pack.py": "medcopilot/scripts/release_pack.py",
    "load_check.py": "medcopilot/scripts/load_check.py",
}


# ------------------------------------------------------- 纯函数契约


class TestQuantiles:
    def test_quantiles_basic(self):
        from medcopilot.scripts.load_check import quantiles

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        p50, p95 = quantiles(values, [0.5, 0.95])
        assert p50 == pytest.approx(5.5)
        assert p95 == pytest.approx(9.55)

    def test_quantiles_empty_raises(self):
        from medcopilot.scripts.load_check import quantiles

        with pytest.raises(ValueError):
            quantiles([], [0.5])

    def test_quantiles_out_of_range_raises(self):
        from medcopilot.scripts.load_check import quantiles

        with pytest.raises(ValueError):
            quantiles([1.0], [1.5])


class TestGateAggregation:
    def test_all_required_pass_overall_pass(self):
        from medcopilot.scripts.release_pack import aggregate_gates

        gates = [
            {"name": "a", "required": True, "status": "pass"},
            {"name": "b", "required": False, "status": "expected_fail"},
        ]
        assert aggregate_gates(gates) == "pass"

    def test_required_fail_overall_fail(self):
        from medcopilot.scripts.release_pack import aggregate_gates

        gates = [{"name": "a", "required": True, "status": "fail"}]
        assert aggregate_gates(gates) == "fail"

    def test_expected_fail_does_not_block(self):
        from medcopilot.scripts.release_pack import aggregate_gates

        gates = [
            {"name": "unit", "required": True, "status": "pass"},
            {"name": "rag_quality", "required": False, "status": "expected_fail"},
        ]
        assert aggregate_gates(gates) == "pass"

    def test_unknown_status_raises(self):
        from medcopilot.scripts.release_pack import aggregate_gates

        with pytest.raises(ValueError):
            aggregate_gates([{"name": "a", "required": True, "status": "meh"}])

    def test_required_gate_failure_maps_to_fail_not_expected_status(self):
        """必须门失败必须是 fail——绝不被 expected_status 洗白成 pass。"""
        from medcopilot.scripts.release_pack import gate_status

        assert gate_status(0, required=True, expected_status="pass") == "pass"
        assert gate_status(1, required=True, expected_status="pass") == "fail"

    def test_optional_gate_failure_maps_to_expected_fail(self):
        from medcopilot.scripts.release_pack import gate_status

        assert gate_status(1, required=False, expected_status="expected_fail") == "expected_fail"

    def test_run_gate_required_failure_is_not_laundered(self, monkeypatch):
        """端到端钉死：必须门子进程退出非零 ⇒ status=fail 且聚合为 fail。"""
        from medcopilot.scripts import release_pack

        monkeypatch.setitem(
            release_pack.GATES,
            "_test_required_fail",
            {
                "command": [sys.executable, "-c", "raise SystemExit(1)"],
                "required": True,
                "expected_status": "pass",
                "env": {},
            },
        )
        gate = release_pack._run_gate("_test_required_fail")
        assert gate["exit_code"] == 1
        assert gate["status"] == "fail"
        assert release_pack.aggregate_gates([gate]) == "fail"


class TestCrashClassification:
    """W8 退出码可分辨化：意外退出码必须显示为 crash 并阻断整体，绝不混入 fail/expected_fail。

    缺省（未声明 expected_exit_codes）保持既有语义：任何非零都是普通失败；
    显式声明后，集合外的退出码是工具崩溃——崩溃的工具不产出证据，无论
    必须与否一律阻断 overall。
    """

    def test_gate_status_defaults_keep_legacy_semantics(self):
        from medcopilot.scripts.release_pack import gate_status

        assert gate_status(0, required=True, expected_status="pass") == "pass"
        assert gate_status(1, required=True, expected_status="pass") == "fail"
        assert gate_status(1, required=False, expected_status="expected_fail") == "expected_fail"

    def test_gate_status_expected_exit_codes_enable_crash_classification(self):
        from medcopilot.scripts.release_pack import gate_status

        codes = [0, 1]
        assert gate_status(1, required=False, expected_status="expected_fail", expected_exit_codes=codes) == (
            "expected_fail"
        )
        assert gate_status(2, required=False, expected_status="expected_fail", expected_exit_codes=codes) == "crash"
        assert gate_status(2, required=True, expected_status="pass", expected_exit_codes=codes) == "crash"

    def test_unexpected_exit_code_blocks_overall_even_for_optional_gate(self, monkeypatch):
        from medcopilot.scripts import release_pack

        monkeypatch.setitem(
            release_pack.GATES,
            "_test_crash",
            {
                "command": [sys.executable, "-c", "raise SystemExit(2)"],
                "required": False,
                "expected_status": "expected_fail",
                "expected_exit_codes": [0],
                "env": {},
            },
        )
        gate = release_pack._run_gate("_test_crash")
        assert gate["exit_code"] == 2
        assert gate["status"] == "crash"
        assert release_pack.aggregate_gates([gate]) == "fail"

    def test_run_gate_records_stderr_tail(self, monkeypatch):
        from medcopilot.scripts import release_pack

        monkeypatch.setitem(
            release_pack.GATES,
            "_test_stderr",
            {
                "command": [
                    sys.executable,
                    "-c",
                    "import sys\nprint('boom-line', file=sys.stderr)\nraise SystemExit(2)",
                ],
                "required": True,
                "expected_status": "pass",
                "expected_exit_codes": [0],
                "env": {},
            },
        )
        gate = release_pack._run_gate("_test_stderr")
        assert gate["status"] == "crash"
        assert "boom-line" in "\n".join(gate["stderr_tail"])

    def test_rag_quality_gate_keeps_designed_fail_with_evidence(self):
        """rag_quality：eval_rag 设计内 FAIL（exit 1）⇒ expected_fail，stdout_tail 保留 FAIL 判据。"""
        from medcopilot.scripts import release_pack

        gate = release_pack._run_gate("rag_quality")
        assert gate["exit_code"] == 1
        assert gate["status"] == "expected_fail"
        assert "FAIL (faithfulness" in "\n".join(gate["stdout_tail"])
        assert release_pack.aggregate_gates([gate]) == "pass"

    def test_crash_is_a_valid_aggregate_status_and_blocks(self):
        from medcopilot.scripts.release_pack import aggregate_gates

        assert aggregate_gates([{"name": "a", "required": False, "status": "crash"}]) == "fail"
        with pytest.raises(ValueError):
            aggregate_gates([{"name": "a", "required": True, "status": "meh"}])


# ------------------------------------------------------- CLI


def _run_cli(script: str, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": ".", "MOCK_LLM": "1"}
    return subprocess.run(
        [sys.executable, SCRIPT_PATHS[script], *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_release_pack_cli_subset_is_incomplete(tmp_path):
    out = tmp_path / "release_pack.json"
    result = _run_cli("release_pack.py", "--only", "policy_gate,golden_gate", "--output", str(out))
    assert result.returncode == 1, result.stdout + result.stderr
    pack = json.loads(out.read_text(encoding="utf-8"))
    assert pack["pack_schema_version"] == 2
    assert pack["overall"] == "incomplete"
    assert pack["partial"] is True
    assert pack["incomplete"] is True
    names = {gate["name"] for gate in pack["gates"]}
    assert names == {"policy_gate", "golden_gate"}
    for gate in pack["gates"]:
        assert gate["exit_code"] == 0
        assert gate["status"] == "pass"
        assert gate["duration_seconds"] >= 0
    assert isinstance(pack["git"], dict)


def test_release_pack_expected_fail_does_not_block(tmp_path):
    out = tmp_path / "release_pack.json"
    result = _run_cli("release_pack.py", "--only", "rag_quality,policy_gate", "--output", str(out))
    assert result.returncode == 1, result.stdout + result.stderr
    pack = json.loads(out.read_text(encoding="utf-8"))
    by_name = {gate["name"]: gate for gate in pack["gates"]}
    assert by_name["rag_quality"]["status"] == "expected_fail"
    assert by_name["policy_gate"]["status"] == "pass"
    assert pack["overall"] == "incomplete"
    assert pack["partial"] is True


def test_complete_opensource_pack_has_ops_gates_and_clinical_block(tmp_path, monkeypatch):
    from medcopilot.scripts import release_pack

    def fake_gate(name, overrides=None):
        spec = release_pack.GATES[name]
        return {"name": name, "required": spec["required"], "status": "expected_fail" if name == "clinical_signoff" else "pass", "exit_code": 4 if name == "clinical_signoff" else 0, "duration_seconds": 0}

    monkeypatch.setattr(release_pack, "_run_gate", fake_gate)
    pack = release_pack.build_pack(list(release_pack.GATES), profile="opensource")
    assert {gate["name"] for gate in pack["gates"]} >= {"recovery_drill", "load_check", "redact_scan"}
    assert pack["overall"] == "engineering_pass_clinical_blocked"
    assert "ops_evidence" in pack


def test_production_profile_requires_full_set_and_never_injects_mock(monkeypatch):
    from medcopilot.scripts import release_pack

    seen = []

    def fake_gate(name, overrides=None):
        seen.append((name, overrides or {}))
        return {"name": name, "required": True, "status": "pass", "exit_code": 0, "duration_seconds": 0}

    monkeypatch.setattr(release_pack, "_run_gate", fake_gate)
    # The blockers read the process environment; pin it so the test does not
    # depend on whether the runner happens to export MOCK_LLM / PROJECT_ENV.
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.delenv("PROJECT_ENV", raising=False)
    selected = list(release_pack.GATES)
    pack = release_pack.build_pack(selected, profile="production")
    assert pack["overall"] == "fail"
    assert "production_environment_required" in pack["profile_blockers"]
    assert "mock_llm_enabled" in pack["profile_blockers"]
    assert all(item[1].get("profile") == "production" for item in seen)
    assert "clinical_signoff" in {item[0] for item in seen}


def test_redact_scan_rejects_secret_and_accepts_clean_json(tmp_path):
    from medcopilot.scripts.redact_scan import scan_path

    clean = tmp_path / "clean.json"
    clean.write_text('{"status":"pass","count":1}', encoding="utf-8")
    assert scan_path(tmp_path) == []
    clean.write_text('{"api_key":"super-secret-value"}', encoding="utf-8")
    assert scan_path(tmp_path)


def test_load_check_cli_reports_percentiles(tmp_path):
    out = tmp_path / "load_report.json"
    result = _run_cli("load_check.py", "--concurrency", "4", "--requests-per-worker", "2",
                      "--output", str(out), timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["total_requests"] == 8
    assert report["errors"] == 0
    assert report["latency_p50_seconds"] >= 0
    assert report["latency_p95_seconds"] >= report["latency_p50_seconds"]
    assert report["ready"] is True
