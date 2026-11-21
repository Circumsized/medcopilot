"""W6.2 策略工件签名与临床签核双态契约。

支持"无医师"(status=none,/health 如实暴露 false,release_pack 记 expected_fail)
与"有医师"(HMAC 签名 + approved 签核,--require-signoff 转发布阻断门)两种运行态。
签名密钥绝不入库(POLICY_SIGNING_KEY 环境变量注入)。
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from medcopilot.safety.policy import (
    canonical_sha256,
    describe_health,
    load_policy,
    sign_policy_payload,
    verify_signatures,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO_ROOT / "medcopilot" / "datasets" / "policies" / "safety_policy_v1.json"
SIGNING_KEY = "unit-test-signing-key"


def _run_sign(*args: str, key: str | None = SIGNING_KEY) -> subprocess.CompletedProcess:
    env = {**os.environ, "MOCK_LLM": "1", "PYTHONPATH": "."}
    if key is not None:
        env["POLICY_SIGNING_KEY"] = key
    return subprocess.run(
        [sys.executable, "medcopilot/scripts/policy_sign.py", *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )


def _run_check(*args: str, key: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "MOCK_LLM": "1", "PYTHONPATH": "."}
    if key is not None:
        env["POLICY_SIGNING_KEY"] = key
    return subprocess.run(
        [sys.executable, "medcopilot/scripts/policy_check.py", *args],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180,
    )


# ------------------------------------------------------------ 无医师运行态(现状)


def test_repo_artifact_starts_unsigned_and_unsignoff():
    policy = load_policy(POLICY_PATH)
    assert policy.signatures == ()
    assert policy.clinical_signoff_approved is False
    health = describe_health(str(POLICY_PATH))
    assert health["policy_signature_present"] is False
    assert health["clinical_signoff"] is False


def test_require_signature_and_signoff_fail_without_them(tmp_path):
    assert _run_check("--require-signature", key=SIGNING_KEY).returncode == 3
    assert _run_check("--require-signoff").returncode == 4
    assert _run_check("--require-signature").returncode == 3  # 无密钥也如实报 3


# ------------------------------------------------------------ 有医师运行态


def test_sign_and_verify_roundtrip(tmp_path):
    out = tmp_path / "signed.json"
    doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    policy_path = str(out)

    assert _run_sign("sign", "--policy", policy_path, "--signed-by", "维护者A").returncode == 0
    assert _run_sign("verify", "--policy", policy_path).returncode == 0
    # 换密钥 ⇒ 验签失败
    assert _run_sign("verify", "--policy", policy_path, key="wrong-key").returncode == 3

    policy = load_policy(policy_path)
    assert verify_signatures(policy, SIGNING_KEY) is True
    assert policy.signatures[0]["covers"]["content_sha256"] == policy.content_sha256


def test_tampered_content_breaks_signature(tmp_path):
    out = tmp_path / "signed.json"
    doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    policy_path = str(out)
    assert _run_sign("sign", "--policy", policy_path).returncode == 0

    tampered = json.loads(out.read_text(encoding="utf-8"))
    tampered["content"]["grounding_stop_words"] = []
    out.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    # 指纹门先失败(内容改了没重算指纹)
    with pytest.raises(Exception):
        load_policy(out)


def test_signoff_lifecycle_none_to_approved(tmp_path):
    out = tmp_path / "signed.json"
    doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    policy_path = str(out)

    assert _run_sign("approve-signoff", "--policy", policy_path,
                     "--reviewer", "张药师", "--credentials", "执业药师证-XXXX",
                     "--scope", "急救词表与剂量话术临床适当性").returncode == 0
    policy = load_policy(policy_path)
    assert policy.clinical_signoff_approved is True
    assert policy.clinical_signoff["covers_sha256"] == policy.content_sha256
    health = describe_health(policy_path)
    assert health["clinical_signoff"] is True
    # 签核后阻断门通过
    assert _run_check("--require-signoff", "--policy", policy_path).returncode == 0


def test_approved_signoff_requires_reviewer_and_current_fingerprint(tmp_path):
    doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    sha = canonical_sha256(doc["content"])
    bad = copy.deepcopy(doc)
    bad["clinical_signoff"] = {"status": "approved", "reviewer": "", "credentials_ref": "",
                               "date": "2026-10-04", "covers_sha256": sha, "scope": "s"}
    from medcopilot.safety.policy import PolicyArtifactError

    with pytest.raises(PolicyArtifactError):
        load_policy(_write(tmp_path, bad))

    stale = copy.deepcopy(doc)
    stale["clinical_signoff"] = {"status": "approved", "reviewer": "张药师",
                                 "credentials_ref": "ref", "date": "2026-10-04",
                                 "covers_sha256": "0" * 64, "scope": "s"}
    with pytest.raises(PolicyArtifactError):
        load_policy(_write(tmp_path, stale))


def test_pending_signoff_requires_reviewer(tmp_path):
    doc = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    doc["clinical_signoff"] = {"status": "pending", "reviewer": "", "credentials_ref": "",
                               "date": "", "covers_sha256": "", "scope": "", "notes": ""}
    from medcopilot.safety.policy import PolicyArtifactError

    with pytest.raises(PolicyArtifactError):
        load_policy(_write(tmp_path, doc))


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / f"policy-{abs(hash(json.dumps(doc, sort_keys=True)))}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def test_release_pack_carries_new_gates():
    """release_pack 含 emergency_eval(必须门)与 clinical_signoff(诚实 advisory 门)。"""
    from medcopilot.scripts.release_pack import GATES, aggregate_gates

    assert "emergency_eval" in GATES and GATES["emergency_eval"]["required"] is True
    assert "clinical_signoff" in GATES
    assert GATES["clinical_signoff"]["expected_status"] == "expected_fail"
    assert GATES["clinical_signoff"]["required"] is False
    # advisory 门失败不阻断整体(expected_fail 语义)
    assert aggregate_gates([
        {"name": "clinical_signoff", "status": "expected_fail", "required": False},
    ]) == "pass"
    assert aggregate_gates([
        {"name": "emergency_eval", "status": "fail", "required": True},
    ]) == "fail"


def test_hmac_payload_is_deterministic():
    content = {"a": 1}
    assert sign_policy_payload(3, content, "k") == sign_policy_payload(3, content, "k")
    assert sign_policy_payload(3, content, "k") != sign_policy_payload(3, content, "k2")
    assert sign_policy_payload(3, content, "k") != sign_policy_payload(4, content, "k")
