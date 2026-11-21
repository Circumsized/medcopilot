"""医疗安全策略的版本化与评审门（Slice 5）。

本包只做审计、版本化与一致性校验；实际门控逻辑与执行源始终在
`workflow/safety_guard.py`、`agents/triage.py` 与 `AppConfig` 的代码常量中。
策略工件损坏、缺失或未通过评审时，健康端点与 `medcopilot/scripts/policy_check.py`
会如实报告失败，但任何实际安全门都绝不因此削弱（fail-safe）。
"""

from medcopilot.safety.policy import (
    PolicyArtifactError,
    PolicyReviewError,
    SafetyPolicy,
    canonical_sha256,
    check_code_equivalence,
    describe_health,
    load_policy,
)

__all__ = [
    "PolicyArtifactError",
    "PolicyReviewError",
    "SafetyPolicy",
    "canonical_sha256",
    "check_code_equivalence",
    "describe_health",
    "load_policy",
]
