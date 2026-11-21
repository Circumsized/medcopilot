"""Release evidence wiring: every evidence file build_pack reads must have a writer,
present-but-failing evidence must block production, and code gates must not be run
inside a production environment they were never designed for."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from medcopilot.scripts import release_pack
from medcopilot.scripts.redact_scan import main as redact_main

SECRET = "sk-live-0123456789abcdef"


def test_redact_scan_writes_summary_without_leaking_finding_text(tmp_path):
    (tmp_path / "scan").mkdir()
    (tmp_path / "scan" / "report.json").write_text(json.dumps({"api_key": SECRET}), encoding="utf-8")
    out = tmp_path / "redact_scan.json"
    assert redact_main([str(tmp_path / "scan"), "--output", str(out)]) == 1
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["all_passed"] is False and summary["findings_count"] >= 1
    assert SECRET not in out.read_text(encoding="utf-8")


def test_redact_scan_clean_directory_writes_passing_summary(tmp_path):
    (tmp_path / "scan").mkdir()
    (tmp_path / "scan" / "ok.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
    out = tmp_path / "redact_scan.json"
    assert redact_main([str(tmp_path / "scan"), "--output", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "report_schema_version": 1, "all_passed": True, "findings_count": 0,
    }


def test_redact_scan_gate_writes_the_file_build_pack_reads():
    command = release_pack.GATES["redact_scan"]["command"]
    assert command[command.index("--output") + 1] == "outputs/redact_scan.json"


def _production_pack(monkeypatch, tmp_path, *, recovery, load, redact):
    monkeypatch.setenv("PROJECT_ENV", "production")
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setattr(
        release_pack, "_run_gate",
        lambda name, overrides=None: {
            "name": name, "required": True, "status": "pass", "exit_code": 0, "duration_seconds": 0,
        },
    )
    paths = {}
    for name, body in (("recovery_drill", recovery), ("load_check", load), ("redact_scan", redact)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        paths[name] = path
    return release_pack.build_pack(list(release_pack.GATES), profile="production", evidence_paths=paths)


GOOD = {
    "recovery": {"all_passed": True},
    "load": {"ready": True, "errors": 0},
    "redact": {"all_passed": True, "findings_count": 0},
}


def test_production_passes_only_when_all_evidence_is_present_and_green(monkeypatch, tmp_path):
    pack = _production_pack(monkeypatch, tmp_path, **GOOD)
    assert pack["profile_blockers"] == []


@pytest.mark.parametrize(
    "key,body,blocker",
    [
        ("recovery", {"all_passed": False}, "failed_recovery_drill_evidence"),
        ("load", {"ready": True, "errors": 3}, "failed_load_check_evidence"),
        ("load", {"ready": False, "errors": 0}, "failed_load_check_evidence"),
        ("redact", {"all_passed": False, "findings_count": 2}, "failed_redact_scan_evidence"),
    ],
)
def test_production_blocks_on_present_but_failing_evidence(monkeypatch, tmp_path, key, body, blocker):
    pack = _production_pack(monkeypatch, tmp_path, **{**GOOD, key: body})
    assert blocker in pack["profile_blockers"]
    assert pack["overall"] == "fail"


def test_gates_are_never_forced_into_production_environment(monkeypatch):
    """load_check/recovery_drill/pytest need a dev env (they start servers without
    ADMIN_API_KEY); forcing PROJECT_ENV=production into them makes every one fail.
    Production readiness is judged from the caller's env via profile_blockers."""
    captured = {}

    def fake_run(command, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(release_pack.subprocess, "run", fake_run)
    monkeypatch.setitem(
        release_pack.GATES, "_probe",
        {"command": [sys.executable, "probe.py"], "required": True,
         "expected_status": "pass", "env": {"MOCK_LLM": "1"}},
    )
    monkeypatch.setenv("PROJECT_ENV", "development")
    release_pack._run_gate("_probe", {"profile": "production"})
    assert captured["env"]["PROJECT_ENV"] == "development"
    assert captured["env"]["MOCK_LLM"] == "1"


def test_load_check_is_independent_of_the_callers_environment(tmp_path):
    """conftest.py sets PROJECT_ENV=development for pytest, which masked this: from a
    plain shell/CI step PROJECT_ENV is unset, W1.3 default-deny answers every read 503,
    and the rate limiter would throttle 40 requests from one address."""
    import os
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items()
           if k not in {"PROJECT_ENV", "MED_RATE_LIMIT_ENABLED", "ADMIN_API_KEY"}}
    env.update({"PYTHONPATH": ".", "MOCK_LLM": "1"})
    out = tmp_path / "load.json"
    completed = subprocess.run(
        [sys.executable, "medcopilot/scripts/load_check.py", "--concurrency", "4",
         "--requests-per-worker", "6", "--output", str(out)],
        cwd=repo, env=env, capture_output=True, text=True, timeout=300,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["total_requests"] == 24
    assert report["errors"] == 0, completed.stdout + completed.stderr
    assert completed.returncode == 0


def test_production_unsigned_policy_is_a_failure_not_a_tool_crash(monkeypatch):
    """Exit 3/4 are the documented "no signature"/"no signoff" outcomes. Reporting them
    as `crash` tells the reader the tool broke, which misstates the evidence."""
    seen = {}

    def fake_run(name, overrides=None):
        seen[name] = overrides or {}
        return {"name": name, "required": True, "status": "pass", "exit_code": 0, "duration_seconds": 0}

    monkeypatch.setattr(release_pack, "_run_gate", fake_run)
    release_pack.build_pack(["policy_signature", "clinical_signoff"], profile="production", evidence_paths={})
    assert 3 in seen["policy_signature"]["expected_exit_codes"]
    assert 4 in seen["clinical_signoff"]["expected_exit_codes"]
    assert release_pack.gate_status(3, True, "pass", seen["policy_signature"]["expected_exit_codes"]) == "fail"
    assert release_pack.gate_status(4, True, "pass", seen["clinical_signoff"]["expected_exit_codes"]) == "fail"
    assert release_pack.gate_status(2, True, "pass", seen["clinical_signoff"]["expected_exit_codes"]) == "crash"
