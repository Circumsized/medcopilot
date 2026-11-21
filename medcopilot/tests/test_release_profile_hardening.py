"""W5 release-profile and evidence-pack contracts."""
from __future__ import annotations

import json


def test_production_profile_requires_full_gate_set_and_does_not_inject_mock(monkeypatch):
    from medcopilot.scripts import release_pack

    seen = []

    def fake_run(name, overrides=None):
        seen.append((name, overrides or {}))
        return {"name": name, "required": True, "status": "pass", "exit_code": 0, "duration_seconds": 0}

    monkeypatch.setattr(release_pack, "_run_gate", fake_run)
    pack = release_pack.build_pack(["config_mirrors"], profile="production")
    assert pack["overall"] == "fail"  # production also requires live/profile preconditions below
    assert all("env" not in override for _, override in seen)


def test_ops_gates_are_required_in_production_and_evidence_is_summarized(monkeypatch, tmp_path):
    from medcopilot.scripts import release_pack

    monkeypatch.setattr(release_pack, "_run_gate", lambda name, overrides=None: {
        "name": name, "required": True, "status": "pass", "exit_code": 0,
        "duration_seconds": 0,
    })
    recovery = tmp_path / "recovery.json"
    load = tmp_path / "load.json"
    recovery.write_text(json.dumps({"all_passed": True, "scenarios": [{"scenario": "x", "passed": True}]}))
    load.write_text(json.dumps({"ready": True, "errors": 0, "latency_p95_seconds": 0.1}))
    pack = release_pack.build_pack(
        list(release_pack.GATES), profile="opensource",
        evidence_paths={"recovery_drill": recovery, "load_check": load},
    )
    assert pack["ops_evidence"]["recovery_drill"]["all_passed"] is True
    assert pack["ops_evidence"]["load_check"]["errors"] == 0


def test_missing_ops_evidence_is_explicitly_incomplete(tmp_path):
    from medcopilot.scripts import release_pack

    selected = ["config_mirrors"]
    pack = release_pack.build_pack(selected, profile="opensource", evidence_paths={})
    assert pack["overall"] == "incomplete"
    assert pack["ops_evidence"]["recovery_drill"]["status"] == "missing"
    assert pack["ops_evidence"]["load_check"]["status"] == "missing"


def test_production_profile_requires_explicit_production_environment(monkeypatch):
    from medcopilot.scripts import release_pack

    monkeypatch.setenv("PROJECT_ENV", "development")
    monkeypatch.setenv("MOCK_LLM", "1")
    pack = release_pack.build_pack(["config_mirrors"], profile="production", evidence_paths={})
    assert pack["overall"] == "fail"
    assert "production_environment_required" in pack["profile_blockers"]
    assert "mock_llm_enabled" in pack["profile_blockers"]
