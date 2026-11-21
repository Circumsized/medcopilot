"""Slice 5 契约：医疗安全策略版本化与专家复核。

策略工件（datasets/policies/*.json）是运行时安全策略的审计与版本化镜像：
- 内容指纹 = canonical JSON 的 sha256，篡改内容必然导致指纹不符；
- 生效要求存在覆盖当前指纹的 approved 评审记录；
- 工件损坏/缺失绝不削弱代码常量里的实际门控（fail-safe）；
- 工件与代码常量必须逐项等价（模式与顺序、文本逐字节、阈值）。

运行时门控的执行源始终是代码常量；本套件锁定"版本化 + 评审门 + 等价守护"，
不测试（也不允许）策略工件反向驱动或削弱门控。
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from medcopilot.agents.triage import TriageAgent
from medcopilot.common.config import AppConfig, config
from medcopilot.safety.policy import (
    PolicyArtifactError,
    PolicyReviewError,
    canonical_sha256,
    check_code_equivalence,
    describe_health,
    load_policy,
)
from medcopilot.workflow.safety_guard import DosageGate, MedicalSafetyGuard

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "medcopilot" / "datasets" / "policies" / "safety_policy_v1.json"


def expected_content() -> dict:
    """从代码常量提取工件应记录的同一组字段（等价基线）。"""
    sentinel = "SENTINEL答案"
    with_evidence = MedicalSafetyGuard.post_process_response(sentinel, has_retrieved_evidence=True)
    without = MedicalSafetyGuard.post_process_response(sentinel, has_retrieved_evidence=False)
    base = with_evidence[len(sentinel):]
    full_without = without[len(sentinel):]
    prefix = full_without[: len(full_without) - len(base)]
    _, alert = MedicalSafetyGuard.check_emergency("怀疑心梗")
    return {
        "policy_version": MedicalSafetyGuard.POLICY_VERSION,
        "emergency_patterns": [p.pattern for p in MedicalSafetyGuard.EMERGENCY_PATTERNS],
        "emergency_alert_text": alert,
        "non_medical_keywords": list(TriageAgent.NON_MEDICAL_KEYWORDS),
        "medical_hints": list(TriageAgent.MEDICAL_HINTS),
        "grounding_stop_words": sorted(MedicalSafetyGuard.STOP_WORDS),
        "disclaimer_base": base,
        "disclaimer_no_evidence_prefix": prefix,
        "evidence_floor_default": AppConfig.model_fields["evidence_floor"].default,
        "groundedness_threshold_default": AppConfig.model_fields["groundedness_threshold"].default,
        "dosage_gate_mode": AppConfig.model_fields["dosage_gate_mode"].default,
        "dosage_referral_text": DosageGate.DOSAGE_REFERRAL_TEXT,
        "sufficiency_min_doc_coverage_default": AppConfig.model_fields["sufficiency_min_doc_coverage"].default,
        "sufficiency_min_union_coverage_default": AppConfig.model_fields["sufficiency_min_union_coverage"].default,
        "sufficiency_dry_run_default": AppConfig.model_fields["sufficiency_dry_run"].default,
        "safety_section_gate_enabled": AppConfig.model_fields["safety_section_gate_enabled"].default,
    }


def approved_reviews(sha: str) -> list[dict]:
    return [{
        "review_id": "unit-test-approved",
        "decision": "approved",
        "covers_sha256": sha,
        "reviewer": "单元测试评审人",
        "date": "2026-09-12",
        "scope": "契约测试夹具",
        "notes": "仅测试用",
    }]


def make_artifact(content: dict | None = None, reviews: list | None = None, drop: tuple = ()) -> dict:
    payload = expected_content() if content is None else content
    sha = canonical_sha256(payload)
    doc = {
        "kind": "medical-safety-policy",
        # 顶层版本必须镜像 content.policy_version(单一事实源的一致性校验)
        "policy_version": payload["policy_version"],
        "content": payload,
        "content_sha256": sha,
        "reviews": approved_reviews(sha) if reviews is None else reviews,
    }
    for key in drop:
        doc.pop(key, None)
    return doc


def write_artifact(tmp_path: Path, doc: dict, name: str = "policy.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------- 工件与等价


class TestRepoArtifact:
    def test_repo_artifact_exists_and_equivalent_to_code(self):
        assert POLICY_PATH.is_file(), "策略工件必须受 Git 跟踪存在于 datasets/policies/"
        policy = load_policy(POLICY_PATH)
        assert policy.policy_version >= 1
        assert check_code_equivalence(policy) == []

    def test_artifact_covers_full_policy_surface(self):
        doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        content = doc["content"]
        # 急救模式必须完整镜像且非空（顺序与代码一致由等价测试保证）
        assert content["emergency_patterns"] == [p.pattern for p in MedicalSafetyGuard.EMERGENCY_PATTERNS]
        assert len(content["emergency_patterns"]) >= 10
        assert content["policy_version"] == MedicalSafetyGuard.POLICY_VERSION
        for key in (
            "emergency_alert_text", "non_medical_keywords", "medical_hints", "grounding_stop_words",
            "disclaimer_base", "disclaimer_no_evidence_prefix",
            "evidence_floor_default", "groundedness_threshold_default",
        ):
            assert key in content, f"工件缺少字段 {key}"

    def test_bootstrap_review_is_honest_about_not_being_clinical(self):
        doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        assert doc["reviews"], "工件必须携带评审记录"
        serialized = json.dumps(doc["reviews"], ensure_ascii=False)
        assert "非临床" in serialized, "bootstrap 评审必须如实声明其非临床专家评审性质"


class TestLoaderValidation:
    def test_valid_temporary_artifact_loads_and_equivalent(self, tmp_path):
        policy = load_policy(write_artifact(tmp_path, make_artifact()))
        assert check_code_equivalence(policy) == []
        assert policy.review_status == "approved"

    def test_missing_file_raises_artifact_error(self, tmp_path):
        with pytest.raises(PolicyArtifactError):
            load_policy(tmp_path / "does-not-exist.json")

    def test_malformed_json_raises_artifact_error(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(PolicyArtifactError):
            load_policy(path)

    def test_unknown_top_level_field_raises_artifact_error(self, tmp_path):
        doc = make_artifact()
        doc["experimental_override"] = {"weaken": True}
        with pytest.raises(PolicyArtifactError):
            load_policy(write_artifact(tmp_path, doc))

    def test_missing_content_sha_raises_artifact_error(self, tmp_path):
        with pytest.raises(PolicyArtifactError):
            load_policy(write_artifact(tmp_path, make_artifact(drop=("content_sha256",))))

    def test_uncompilable_pattern_raises_artifact_error(self, tmp_path):
        content = expected_content()
        content["emergency_patterns"] = content["emergency_patterns"] + ["(未闭合"]
        doc = make_artifact(content)  # 指纹与内容一致，但仍须因正则非法被拒
        with pytest.raises(PolicyArtifactError):
            load_policy(write_artifact(tmp_path, doc))

    def test_empty_reviews_rejected(self, tmp_path):
        doc = make_artifact(reviews=[])
        with pytest.raises(PolicyReviewError):
            load_policy(write_artifact(tmp_path, doc))

    def test_only_pending_reviews_rejected(self, tmp_path):
        doc = make_artifact()
        reviews = copy.deepcopy(doc["reviews"])
        reviews[0]["decision"] = "pending"
        doc["reviews"] = reviews
        with pytest.raises(PolicyReviewError):
            load_policy(write_artifact(tmp_path, doc))

    def test_tampered_content_rejected_by_fingerprint(self, tmp_path):
        doc = make_artifact()
        # 篡改内容但保留旧指纹与旧评审：必须被发现
        doc["content"]["emergency_patterns"] = doc["content"]["emergency_patterns"][1:]
        with pytest.raises(PolicyReviewError):
            load_policy(write_artifact(tmp_path, doc))

    def test_approved_review_covering_old_hash_rejected(self, tmp_path):
        content = expected_content()
        stale_sha = canonical_sha256(content)
        content["grounding_stop_words"] = []  # 掏空停用词表也属内容变更
        doc = {
            "kind": "medical-safety-policy",
            "policy_version": content["policy_version"],
            "content": content,
            "content_sha256": canonical_sha256(content),
            "reviews": approved_reviews(stale_sha),
        }
        with pytest.raises(PolicyReviewError):
            load_policy(write_artifact(tmp_path, doc))


class TestEquivalenceGuard:
    def test_removed_emergency_pattern_detected_as_drift(self, tmp_path):
        content = expected_content()
        content["emergency_patterns"] = content["emergency_patterns"][1:]  # 少一条急救模式
        policy = load_policy(write_artifact(tmp_path, make_artifact(content)))
        drift = check_code_equivalence(policy)
        assert any("emergency_patterns" in item for item in drift)

    def test_drifted_threshold_detected(self, tmp_path):
        content = expected_content()
        content["groundedness_threshold_default"] = 0.0  # 削弱 grounding 门
        policy = load_policy(write_artifact(tmp_path, make_artifact(content)))
        drift = check_code_equivalence(policy)
        assert any("groundedness_threshold_default" in item for item in drift)

    def test_reordered_patterns_detected(self, tmp_path):
        content = expected_content()
        content["emergency_patterns"] = list(reversed(content["emergency_patterns"]))
        policy = load_policy(write_artifact(tmp_path, make_artifact(content)))
        assert any("emergency_patterns" in item for item in check_code_equivalence(policy))


# ------------------------------------------------------------- fail-safe 面


class TestFailSafe:
    def test_describe_health_valid_artifact(self, tmp_path):
        info = describe_health(str(write_artifact(tmp_path, make_artifact())))
        assert info == {
            "policy_version": MedicalSafetyGuard.POLICY_VERSION,
            "policy_matches_code": True,
            "policy_review_status": "approved",
            "policy_signature_present": False,
            "clinical_signoff": False,
        }

    def test_describe_health_missing_artifact_is_invalid_not_fatal(self, tmp_path):
        info = describe_health(str(tmp_path / "missing.json"))
        assert info["policy_version"] is None
        assert info["policy_matches_code"] is False
        assert info["policy_review_status"] == "invalid"

    def test_health_endpoint_exposes_policy_and_triage_survives_missing_artifact(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        from medcopilot.api import server

        configured = SimpleNamespace(**server.config.model_dump())
        configured.safety_policy_path = str(tmp_path / "missing.json")
        monkeypatch.setattr(server, "config", configured)
        client = TestClient(server.create_app())

        health = client.get("/health").json()
        assert health["policy_version"] is None
        assert health["policy_matches_code"] is False
        assert health["policy_review_status"] == "invalid"
        # 工件缺失绝不能影响急救拦截与既有响应字段
        response = client.post("/consultations", json={"query": "我爸突发剧烈胸痛，怀疑是心梗！"})
        assert response.status_code == 200
        body = response.json()
        assert body["is_emergency"] is True
        assert {"answer", "sources", "is_emergency", "intent", "abstained", "session_id"} <= set(body)

    def test_health_endpoint_reports_approved_policy(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from fastapi.testclient import TestClient

        from medcopilot.api import server

        configured = SimpleNamespace(**server.config.model_dump())
        configured.safety_policy_path = str(write_artifact(tmp_path, make_artifact()))
        monkeypatch.setattr(server, "config", configured)
        health = TestClient(server.create_app()).get("/health").json()
        assert health["policy_version"] == MedicalSafetyGuard.POLICY_VERSION
        assert health["policy_matches_code"] is True
        assert health["policy_review_status"] == "approved"

    def test_default_config_points_at_repo_artifact(self):
        # 默认配置即启用仓库工件，健康端点默认报告已评审通过的镜像
        assert config.safety_policy_path  # 非空
        info = describe_health(config.safety_policy_path)
        assert info["policy_review_status"] == "approved"
        assert info["policy_matches_code"] is True


# ------------------------------------------------------------- CLI


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "MOCK_LLM": "1", "PYTHONPATH": "."}
    return subprocess.run(
        [sys.executable, "medcopilot/scripts/policy_check.py", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestCli:
    def test_cli_passes_on_repo_artifact(self):
        result = _run_cli()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "PASS" in result.stdout

    def test_cli_tampered_copy_exits_one(self, tmp_path):
        doc = make_artifact()
        doc["content"]["emergency_alert_text"] = "弱化后的警示文本"  # 指纹不再覆盖
        path = write_artifact(tmp_path, doc)
        result = _run_cli("--policy", str(path))
        assert result.returncode == 1, result.stdout + result.stderr

    def test_cli_drift_copy_exits_one(self, tmp_path):
        content = expected_content()
        content["evidence_floor_default"] = 0.0
        path = write_artifact(tmp_path, make_artifact(content))  # 指纹一致但偏离代码
        result = _run_cli("--policy", str(path))
        assert result.returncode == 1, result.stdout + result.stderr

    def test_cli_broken_artifact_exits_two(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        result = _run_cli("--policy", str(path))
        assert result.returncode == 2, result.stdout + result.stderr
