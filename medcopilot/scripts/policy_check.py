"""医疗安全策略工件门禁（Slice 5 + W6.2）。

退出码：0 = 工件加载通过评审门且与代码常量逐字节一致；
1 = 评审门失败（内容指纹不符 / 无覆盖当前指纹的 approved 评审）或工件
内容已偏离代码常量（漂移）；2 = 工件结构性非法（缺失、坏 JSON、未知
字段、正则不可编译等）；3 = ``--require-signature`` 开启且签名缺失/无效；
4 = ``--require-signoff`` 开启且无 approved 临床签核。

用法::

    python medcopilot/scripts/policy_check.py [--policy PATH]
        [--require-signature] [--require-signoff]

``--require-signature`` 需要 ``POLICY_SIGNING_KEY`` 环境变量(HMAC 密钥)。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medcopilot.common.config import config
from medcopilot.safety.policy import (
    PolicyArtifactError,
    PolicyReviewError,
    check_code_equivalence,
    load_policy,
    verify_signatures,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="medical safety policy artifact gate")
    parser.add_argument(
        "--policy",
        default=config.safety_policy_path,
        help="path to the versioned policy artifact (default: SAFETY_POLICY_PATH)",
    )
    parser.add_argument(
        "--require-signature", action="store_true",
        help="fail (exit 3) unless a valid maintainer signature covers the artifact",
    )
    parser.add_argument(
        "--require-signoff", action="store_true",
        help="fail (exit 4) unless an approved clinical signoff covers the artifact",
    )
    args = parser.parse_args(argv)

    try:
        policy = load_policy(args.policy)
    except PolicyArtifactError as exc:
        print(f"FAIL: invalid policy artifact: {exc}", file=sys.stderr)
        return 2
    except PolicyReviewError as exc:
        print(f"FAIL: policy review gate failed: {exc}", file=sys.stderr)
        return 1

    drift = check_code_equivalence(policy)
    if drift:
        print("FAIL: policy artifact drifted from code constants:", file=sys.stderr)
        for item in drift:
            print(f"  - {item}", file=sys.stderr)
        return 1

    if args.require_signature:
        signing_key = os.environ.get("POLICY_SIGNING_KEY", "")
        if not signing_key:
            print("FAIL: --require-signature needs POLICY_SIGNING_KEY in the environment", file=sys.stderr)
            return 3
        if not policy.signatures or not verify_signatures(policy, signing_key):
            print("FAIL: no valid maintainer signature covers this artifact", file=sys.stderr)
            return 3

    if args.require_signoff and not policy.clinical_signoff_approved:
        print(
            "FAIL: no approved clinical signoff covers this artifact "
            "(clinical_signoff.status != approved or covers an older fingerprint)",
            file=sys.stderr,
        )
        return 4

    extras = []
    if policy.signatures:
        extras.append("signed")
    if policy.clinical_signoff_approved:
        extras.append("clinically signed off")
    suffix = f" ({', '.join(extras)})" if extras else ""
    print(
        f"PASS: policy v{policy.policy_version} carries an approved review covering "
        f"sha256:{policy.content_sha256[:12]} and is byte-identical to code constants{suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
