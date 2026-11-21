"""Slice 8 契约测试：本项目运维脚本单机化（LLM 探针 / 恢复演练）。

诚实性是本切片核心：LLM 探针在 mock/未配置时如实报 not-configured；
恢复演练把"已知故障→期望行为"的表格变成可重复执行的自动化。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCRIPT_PATHS = {
    "llm_probe.py": "medcopilot/scripts/llm_probe.py",
    "recovery_drill.py": "medcopilot/scripts/recovery_drill.py",
}


def _run_cli(script: str, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "PYTHONPATH": ".", "MOCK_LLM": "1"}
    return subprocess.run(
        [sys.executable, SCRIPT_PATHS[script], *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )


# ------------------------------------------------------- LLM 探针


class TestLLMProbe:
    def test_mock_or_unconfigured_reports_not_configured(self, tmp_path):
        out = tmp_path / "probe.json"
        result = _run_cli("llm_probe.py", "--output", str(out))
        assert result.returncode == 2
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["status"] == "not-configured"

    def test_probe_success_path_with_injected_transport(self, monkeypatch, tmp_path):
        import httpx as httpx_module

        from medcopilot.scripts import llm_probe

        def fake_transport(request):
            return httpx_module.Response(
                200, json={"choices": [{"message": {"content": "pong"}}]}, request=request,
            )

        result = llm_probe.probe(
            provider="openai_compatible",
            api_base="https://llm.test/v1",
            api_key="probe" + "-key",
            model="probe-model",
            timeout_seconds=5.0,
            transport=httpx_module.MockTransport(fake_transport),
        )
        assert result["status"] == "ok"
        assert result["latency_seconds"] >= 0
        assert "probe-key" not in json.dumps(result), "凭据不得出现在诊断输出"

    def test_probe_auth_failure_maps_to_auth_error(self, monkeypatch, tmp_path):
        import httpx as httpx_module

        from medcopilot.scripts import llm_probe

        def fake_transport(request):
            return httpx_module.Response(401, json={"error": "bad key"}, request=request)

        result = llm_probe.probe(
            provider="openai_compatible",
            api_base="https://llm.test/v1",
            api_key="probe" + "-key",
            model="probe-model",
            timeout_seconds=5.0,
            transport=httpx_module.MockTransport(fake_transport),
        )
        assert result["status"] == "auth-error"
        assert result["http_status"] == 401

    def test_probe_invalid_url_maps_to_unreachable(self):
        from medcopilot.scripts import llm_probe

        result = llm_probe.probe(
            provider="openai_compatible",
            api_base="http://localhost:notaport/v1",
            api_key="probe" + "-key",
            model="probe-model",
            timeout_seconds=1.0,
        )
        assert result["status"] == "unreachable"
        assert "probe-key" not in json.dumps(result)


# ------------------------------------------------------- 恢复演练


class TestRecoveryDrill:
    def test_drill_all_scenarios_pass(self, tmp_path):
        out = tmp_path / "recovery_drill.json"
        result = _run_cli("recovery_drill.py", "--output", str(out))
        assert result.returncode == 0, result.stdout + result.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        scenarios = {item["scenario"]: item for item in report["scenarios"]}
        assert set(scenarios) == {
            "corpus_corrupt_lines_degrade_not_fail",
            "index_build_failure_keeps_previous_manifest",
            "unknown_session_404_but_emergency_issued",
            "broken_policy_artifact_failsafe",
            "unrecognized_env_fails_closed",
            "audit_chain_tamper_detected",
            # W5.5 backup/RPO-RTO scenarios.
            "backup_restore_recovers_serving_state",
            "backup_tamper_fails_closed",
        }
        for scenario, item in scenarios.items():
            assert item["passed"] is True, f"{scenario}: {item.get('detail')}"
        # 场景 2 必须演练真回滚：先有成功 rebuild 建立的 manifest，再让构建失败，
        # 证明旧 manifest/encoder 原样维持且 consult 仍可用。若退化成"从未成功过
        # 的 500"，manifest_unchanged 会是假真值（两侧都为 None）。
        index_failure = scenarios["index_build_failure_keeps_previous_manifest"]
        assert index_failure["detail"]["first_rebuild_status"] == 200
        assert index_failure["detail"]["rebuild_status"] == 500
        assert index_failure["detail"]["manifest_unchanged"] is True
        assert index_failure["detail"]["digest_recorded"] is True
        assert index_failure["detail"]["consult_status"] == 200

    def test_drill_records_contract_scenario_id_when_scenario_raises(self, monkeypatch):
        from medcopilot.scripts import recovery_drill

        def boom(_tmp):
            raise RuntimeError("injected drill crash")

        specs = list(recovery_drill.SCENARIO_SPECS)
        specs[0] = (specs[0][0], boom)
        monkeypatch.setattr(recovery_drill, "SCENARIO_SPECS", tuple(specs))
        report = recovery_drill.run_drill()
        assert report["all_passed"] is False
        scenarios = {item["scenario"]: item for item in report["scenarios"]}
        assert len(scenarios) == 8
        first = scenarios["corpus_corrupt_lines_degrade_not_fail"]
        assert first["passed"] is False
        assert first["detail"]["error_type"] == "RuntimeError"
