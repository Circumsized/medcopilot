"""医疗安全策略的版本化与专家复核（Slice 5）。

策略工件（Git 跟踪的 JSON 文件）是运行时安全策略的**审计与版本化镜像**，
而**不是**门控的执行源。执行源永远是 `MedicalSafetyGuard` / `TriageAgent` /
`AppConfig` 里的代码常量；工件损坏、缺失或评审未通过只会让健康检查与
`medcopilot/scripts/policy_check.py` 门禁如实报错，绝不削弱任何实际安全门（fail-safe）。

版本化与评审机制：
- 内容指纹 = 对 ``content`` 做 canonical JSON（``sort_keys``、UTF-8）后的 sha256；
  任何内容篡改都会导致重算指纹与 ``content_sha256`` 不符。
- 工件生效要求存在至少一条 ``decision == "approved"`` 且 ``covers_sha256``
  等于当前内容指纹的评审记录；内容有改动而未追加新的 approved 记录即失效。
- ``check_code_equivalence`` 逐项比对工件与代码常量（模式串与顺序、文本逐字节、
  关键词集合、阈值默认值），捕获"改代码不改工件"或反向的漂移。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medcopilot.common.logger import logger

POLICY_KIND = "medical-safety-policy"
# 必填顶层字段(v1 起即强制);signatures/clinical_signoff 为 W6.2 可选治理字段。
_REQUIRED_TOP_KEYS = {"kind", "policy_version", "content", "content_sha256", "reviews"}
_ALLOWED_TOP_KEYS = _REQUIRED_TOP_KEYS | {
    # W6.2:维护者签名(HMAC/GPG)与临床签核双态(none|pending|approved)。
    "signatures", "clinical_signoff",
}
_ALLOWED_SIGNATURE_KEYS = {"alg", "key_id", "value", "signed_by", "date", "covers"}
_ALLOWED_SIGNOFF_STATUSES = {"none", "pending", "approved"}
_ALLOWED_CONTENT_KEYS = {
    "policy_version",
    "emergency_patterns",
    "emergency_alert_text",
    "non_medical_keywords",
    "medical_hints",
    "grounding_stop_words",
    "disclaimer_base",
    "disclaimer_no_evidence_prefix",
    "evidence_floor_default",
    "groundedness_threshold_default",
    "dosage_gate_mode",
    "dosage_referral_text",
    "sufficiency_min_doc_coverage_default",
    "sufficiency_min_union_coverage_default",
    "sufficiency_dry_run_default",
    "safety_section_gate_enabled",
}
_ALLOWED_REVIEW_KEYS = {"review_id", "decision", "covers_sha256", "reviewer", "date", "scope", "notes"}
_ALLOWED_DECISIONS = {"approved", "pending", "rejected"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PolicyArtifactError(ValueError):
    """工件结构性非法：文件缺失/不可读、非法 JSON、未知字段、正则不可编译等。"""


class PolicyReviewError(ValueError):
    """工件结构合法但未通过评审门：内容指纹不符，或缺少覆盖当前指纹的 approved 评审。"""


def canonical_sha256(content: dict[str, Any]) -> str:
    """对策略内容做 canonical JSON 编码后的 sha256（与工件生成端一致的序列化）。"""
    payload = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SafetyPolicy:
    policy_version: int
    content: dict[str, Any]
    content_sha256: str
    reviews: tuple[dict[str, Any], ...] = ()
    signatures: tuple[dict[str, Any], ...] = ()
    clinical_signoff: dict[str, Any] | None = None

    @property
    def review_status(self) -> str:
        # load_policy 只放行存在覆盖当前指纹的 approved 评审记录的工件，
        # 因此成功加载的策略评审状态恒为 approved。
        return "approved"

    @property
    def clinical_signoff_approved(self) -> bool:
        """资质评审人签核是否生效(approved 且覆盖当前指纹)。无签核 ⇒ False(如实暴露)。"""
        signoff = self.clinical_signoff or {}
        return signoff.get("status") == "approved" and signoff.get("covers_sha256") == self.content_sha256


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise PolicyArtifactError(message)


def _validate_structural(doc: Any) -> tuple[int, dict[str, Any], str, list[dict[str, Any]]]:
    _require(isinstance(doc, dict), "policy artifact top level must be an object")
    unknown = set(doc) - _ALLOWED_TOP_KEYS
    _require(not unknown, f"unknown top-level fields: {sorted(unknown)}")
    for key in _REQUIRED_TOP_KEYS:
        _require(key in doc, f"missing required field {key!r}")

    _require(doc["kind"] == POLICY_KIND, f"kind must be {POLICY_KIND!r}")
    version = doc["policy_version"]
    _require(
        isinstance(version, int) and not isinstance(version, bool) and version >= 1,
        "policy_version must be a positive integer",
    )

    content = doc["content"]
    _require(isinstance(content, dict), "content must be an object")
    unknown = set(content) - _ALLOWED_CONTENT_KEYS
    _require(not unknown, f"unknown content fields: {sorted(unknown)}")
    missing = _ALLOWED_CONTENT_KEYS - set(content)
    _require(not missing, f"missing content fields: {sorted(missing)}")

    patterns = content["emergency_patterns"]
    _require(
        isinstance(patterns, list) and patterns and all(isinstance(p, str) and p for p in patterns),
        "emergency_patterns must be a non-empty list of non-empty strings",
    )
    _require(len(set(patterns)) == len(patterns), "emergency_patterns must not contain duplicates")
    for index, pattern in enumerate(patterns):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise PolicyArtifactError(f"emergency_patterns[{index}] does not compile: {pattern!r} ({exc})") from exc
    version_mirror = content["policy_version"]
    _require(
        isinstance(version_mirror, int) and not isinstance(version_mirror, bool) and version_mirror >= 1,
        "content.policy_version must be a positive integer",
    )
    _require(
        version_mirror == doc["policy_version"],
        "content.policy_version must match the top-level policy_version",
    )
    _require(
        content["dosage_gate_mode"] in ("redact", "quote", "off"),
        "dosage_gate_mode must be one of redact|quote|off",
    )
    _require(
        isinstance(content["dosage_referral_text"], str) and content["dosage_referral_text"].strip(),
        "dosage_referral_text must be a non-empty string",
    )
    for key in ("emergency_alert_text", "disclaimer_base", "disclaimer_no_evidence_prefix"):
        _require(isinstance(content[key], str) and content[key].strip(), f"{key} must be a non-empty string")
    for key in ("non_medical_keywords", "medical_hints", "grounding_stop_words"):
        value = content[key]
        _require(
            isinstance(value, list) and all(isinstance(x, str) and x for x in value),
            f"{key} must be a list of non-empty strings",
        )
    for key in ("evidence_floor_default", "groundedness_threshold_default",
                "sufficiency_min_doc_coverage_default", "sufficiency_min_union_coverage_default"):
        value = content[key]
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= float(value) <= 1,
            f"{key} must be a number within [0, 1]",
        )
    for key in ("sufficiency_dry_run_default", "safety_section_gate_enabled"):
        _require(isinstance(content[key], bool), f"{key} must be a boolean")

    sha = doc["content_sha256"]
    _require(isinstance(sha, str) and bool(_SHA256_RE.fullmatch(sha)), "content_sha256 must be 64 hex chars")

    reviews = doc["reviews"]
    _require(isinstance(reviews, list), "reviews must be a list")
    seen_ids: set[str] = set()
    for index, record in enumerate(reviews):
        _require(isinstance(record, dict), f"reviews[{index}] must be an object")
        unknown = set(record) - _ALLOWED_REVIEW_KEYS
        _require(not unknown, f"reviews[{index}] has unknown fields: {sorted(unknown)}")
        for key in _ALLOWED_REVIEW_KEYS - {"notes"}:
            _require(isinstance(record.get(key), str) and record[key].strip(), f"reviews[{index}].{key} is required")
        _require(
            isinstance(record.get("covers_sha256"), str) and bool(_SHA256_RE.fullmatch(record["covers_sha256"])),
            f"reviews[{index}].covers_sha256 must be 64 hex chars",
        )
        _require(record["decision"] in _ALLOWED_DECISIONS, f"reviews[{index}].decision must be one of "
                 f"{sorted(_ALLOWED_DECISIONS)}")
        _require(record["review_id"] not in seen_ids, f"reviews[{index}].review_id is duplicated")
        seen_ids.add(record["review_id"])

    # ---- W6.2:签名与临床签核(可选字段;存在时结构性校验) ----
    signatures = doc.get("signatures", [])
    _require(isinstance(signatures, list), "signatures must be a list when present")
    for index, sig in enumerate(signatures):
        _require(isinstance(sig, dict), f"signatures[{index}] must be an object")
        unknown = set(sig) - _ALLOWED_SIGNATURE_KEYS
        _require(not unknown, f"signatures[{index}] has unknown fields: {sorted(unknown)}")
        for key in ("alg", "key_id", "value", "signed_by", "date"):
            _require(isinstance(sig.get(key), str) and sig[key].strip(), f"signatures[{index}].{key} is required")
        covers = sig.get("covers")
        _require(isinstance(covers, dict), f"signatures[{index}].covers must be an object")
        _require(
            covers.get("policy_version") == doc["policy_version"]
            and covers.get("content_sha256") == sha,
            f"signatures[{index}].covers must match the current policy version and fingerprint",
        )

    signoff = doc.get("clinical_signoff", {"status": "none"})
    _require(isinstance(signoff, dict), "clinical_signoff must be an object when present")
    unknown = set(signoff) - {
        "status", "reviewer", "credentials_ref", "date", "covers_sha256", "scope", "notes",
    }
    _require(not unknown, f"clinical_signoff has unknown fields: {sorted(unknown)}")
    status = signoff.get("status")
    _require(status in _ALLOWED_SIGNOFF_STATUSES, f"clinical_signoff.status must be one of {sorted(_ALLOWED_SIGNOFF_STATUSES)}")
    if status == "approved":
        for key in ("reviewer", "credentials_ref", "date", "covers_sha256", "scope"):
            _require(
                isinstance(signoff.get(key), str) and signoff[key].strip(),
                f"clinical_signoff.{key} is required when status is approved",
            )
        _require(
            signoff["covers_sha256"] == sha,
            "clinical_signoff.covers_sha256 must match the current content fingerprint",
        )
    elif status == "pending":
        _require(
            isinstance(signoff.get("reviewer"), str) and signoff["reviewer"].strip(),
            "clinical_signoff.reviewer is required when status is pending",
        )

    return version, content, sha, reviews


def load_policy(path: str | Path) -> SafetyPolicy:
    """加载并校验策略工件。

    结构性问题抛 ``PolicyArtifactError``（CLI 退出码 2）；结构合法但内容指纹
    不符或缺少覆盖当前指纹的 approved 评审时抛 ``PolicyReviewError``（退出码 1）。
    """
    path = Path(path)
    if (
        not path.is_absolute()
        and not path.exists()
        and path.as_posix() == "medcopilot/datasets/policies/safety_policy_v1.json"
    ):
        repo_candidate = Path(__file__).resolve().parents[2] / path
        if repo_candidate.exists():
            path = repo_candidate
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyArtifactError(f"policy artifact unreadable: {path} ({exc})") from exc
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PolicyArtifactError(f"policy artifact is not valid JSON: {path} (line {exc.lineno})") from exc

    version, content, declared_sha, reviews = _validate_structural(doc)

    recomputed = canonical_sha256(content)
    if recomputed != declared_sha:
        raise PolicyReviewError(
            f"policy content does not match its declared fingerprint (recomputed {recomputed}); "
            "the content was modified after signing"
        )
    covering = [r for r in reviews if r["decision"] == "approved" and r["covers_sha256"] == recomputed]
    if not covering:
        raise PolicyReviewError(
            "no approved review covers the current policy fingerprint; "
            "content changes require a newly recorded approved review"
        )
    return SafetyPolicy(
        policy_version=version,
        content=content,
        content_sha256=recomputed,
        reviews=tuple(reviews),
        signatures=tuple(doc.get("signatures", []) or []),
        clinical_signoff=doc.get("clinical_signoff") or {"status": "none"},
    )


def sign_policy_payload(policy_version: int, content: dict[str, Any], signing_key: str) -> str:
    """W6.2 维护者签名:HMAC-SHA256 over {policy_version, content}(canonical JSON)。

    密钥绝不入库(POLICY_SIGNING_KEY 经环境变量/密钥管理注入)。
    """
    import hmac as _hmac

    payload = json.dumps({"policy_version": policy_version, "content": content}, sort_keys=True, ensure_ascii=False)
    return _hmac.new(signing_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_signatures(policy: SafetyPolicy, signing_key: str) -> bool:
    """任一签名能用当前密钥验证且覆盖当前版本与指纹 ⇒ True。"""
    expected = sign_policy_payload(policy.policy_version, policy.content, signing_key)
    return any(
        sig.get("alg") == "HMAC-SHA256"
        and hmac.compare_digest(str(sig.get("value", "")), expected)
        and (sig.get("covers") or {}).get("policy_version") == policy.policy_version
        and (sig.get("covers") or {}).get("content_sha256") == policy.content_sha256
        for sig in policy.signatures
    )


def current_code_content() -> dict[str, Any]:
    """从代码常量提取工件应镜像的字段（等价基线；工件生成与等价检查共用）。"""
    from medcopilot.agents.triage import TriageAgent
    from medcopilot.common.config import AppConfig
    from medcopilot.workflow.safety_guard import DosageGate, MedicalSafetyGuard

    sentinel = "SENTINEL"
    base = MedicalSafetyGuard.post_process_response(sentinel, has_retrieved_evidence=True)[len(sentinel):]
    full_without = MedicalSafetyGuard.post_process_response(sentinel, has_retrieved_evidence=False)[len(sentinel):]
    prefix = full_without[: len(full_without) - len(base)]
    _, alert = MedicalSafetyGuard.check_emergency("怀疑心梗")
    return {
        # 策略版本事实源是代码常量(MedicalSafetyGuard.POLICY_VERSION);工件镜像它,
        # 漂移(改工件不改代码或反向)由 check_code_equivalence 捕获。
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


def check_code_equivalence(policy: SafetyPolicy) -> list[str]:
    """返回工件与代码常量之间的漂移描述列表（空列表表示逐字节一致）。"""
    expected = current_code_content()
    drift: list[str] = []
    for key, want in expected.items():
        got = policy.content.get(key)
        if got != want:
            summary = f"artifact={got!r}"[:120] + f" | code={want!r}"[:120]
            drift.append(f"{key}: {summary}")
    return drift


def describe_health(path: str) -> dict[str, Any]:
    """供 /health 复用的只读状态描述。任何工件问题都不得影响服务本身。"""
    base = {
        "policy_version": None,
        "policy_matches_code": False,
        "policy_review_status": "invalid",
        # W6.2:签名与临床签核如实暴露——"尚无临床签核"必须是机器可读事实。
        "policy_signature_present": False,
        "clinical_signoff": False,
    }
    if not path:
        return {**base, "policy_review_status": "not-configured"}
    try:
        policy = load_policy(path)
    except PolicyReviewError as exc:
        logger.error("Safety policy review gate failed: {}", exc)
        return {**base, "policy_review_status": "not-approved"}
    except Exception as exc:  # 工件缺失/损坏绝不能级联到服务可用性
        logger.error("Safety policy artifact unavailable (serving keeps code-constant gating): {}", exc)
        return base
    try:
        matches = not check_code_equivalence(policy)
    except Exception as exc:
        logger.error("Safety policy equivalence check crashed: {}", exc)
        matches = False
    return {
        "policy_version": policy.policy_version,
        "policy_matches_code": matches,
        "policy_review_status": policy.review_status,
        "policy_signature_present": bool(policy.signatures),
        "clinical_signoff": policy.clinical_signoff_approved,
    }
