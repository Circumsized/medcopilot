"""W5 release-profile and evidence contracts."""
from __future__ import annotations

import json


def test_opensource_complete_pack_is_clinically_blocked_without_signoff():
    from medcopilot.scripts.release_pack import aggregate_gates

    gates = [
        {"name": "unit_tests", "required": True, "status": "pass"},
        {"name": "clinical_signoff", "required": False, "status": "expected_fail"},
    ]
    assert aggregate_gates(gates, profile="opensource") == "engineering_pass_clinical_blocked"


def test_production_profile_never_passes_without_signature_and_signoff():
    from medcopilot.scripts.release_pack import aggregate_gates

    gates = [
        {"name": "unit_tests", "required": True, "status": "pass"},
        {"name": "policy_signature", "required": True, "status": "expected_fail"},
        {"name": "clinical_signoff", "required": True, "status": "expected_fail"},
    ]
    assert aggregate_gates(gates, profile="production") == "fail"


def test_production_profile_requires_full_gate_set(monkeypatch):
    from medcopilot.scripts import release_pack

    # The real "unit_tests" gate is the whole pytest suite; running it from inside
    # the suite re-enters this very test and never terminates. Stub the runner.
    monkeypatch.setattr(
        release_pack, "_run_gate",
        lambda name, overrides=None: {"name": name, "required": True, "status": "pass", "exit_code": 0},
    )
    selected = ["unit_tests", "config_mirrors", "golden_gate", "policy_gate", "emergency_eval"]
    pack = release_pack.build_pack(selected, profile="production")
    assert pack["overall"] == "fail"
    assert pack["partial"] is True


def test_ops_evidence_is_explicit_when_report_is_missing(tmp_path, monkeypatch):
    from medcopilot.scripts import release_pack

    monkeypatch.setattr(release_pack, "REPO_ROOT", tmp_path)
    report = tmp_path / "load.json"
    assert release_pack._ops_evidence(report) == {"status": "missing", "path": str(report)}
    report.write_text(json.dumps({"load_schema_version": 1, "ready": True, "errors": 0}), encoding="utf-8")
    evidence = release_pack._ops_evidence(report)
    assert evidence == {
        "status": "available", "path": str(report), "schema": 1, "all_passed": None,
        "ready": True, "errors": 0, "total_requests": None, "latency_p95_seconds": None,
    }
    report.write_text("{not json", encoding="utf-8")
    assert release_pack._ops_evidence(report)["status"] == "invalid"


def test_gate_specs_include_operational_evidence_gates():
    from medcopilot.scripts.release_pack import GATES

    assert {"recovery_drill", "load_check", "redact_scan"} <= set(GATES)
    assert all(GATES[name]["required"] for name in ("recovery_drill", "load_check", "redact_scan"))
