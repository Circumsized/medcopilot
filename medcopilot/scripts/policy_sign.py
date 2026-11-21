"""策略工件签名与临床签核工具(W6.2)。

支持"无医师"与"有医师"两种运行态:
- 无医师:工件保持 ``clinical_signoff.status="none"``,/health 如实暴露
  ``clinical_signoff=false``;release_pack 以 expected_fail 记录该门(不阻断
  但不可伪装)。产品定位保守(纯科普参考)。
- 有医师:维护者先 ``sign``(HMAC,密钥经 POLICY_SIGNING_KEY 注入,绝不入库),
  资质评审人确认后 ``approve-signoff`` 写入 approved 记录(必须署名、资质引用、
  覆盖当前指纹);此后 ``policy_check --require-signoff`` 转为发布阻断门。

子命令::

    sign       [--policy PATH]                      # 追加/替换维护者 HMAC 签名
    verify     [--policy PATH]                      # 校验签名(需 POLICY_SIGNING_KEY)
    start-signoff  --reviewer NAME --credentials REF [--policy PATH]   # pending
    approve-signoff --reviewer NAME --credentials REF --scope TEXT [--policy PATH]  # approved

退出码:0 成功;2 结构非法;3 签名问题;4 签核问题;1 其它失败。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medcopilot.common.config import config
from medcopilot.common.fsutil import atomic_write_json
from medcopilot.safety.policy import (
    PolicyArtifactError,
    PolicyReviewError,
    _validate_structural,
    canonical_sha256,
    load_policy,
    sign_policy_payload,
    verify_signatures,
)


def _load(path: str) -> tuple[Path, dict]:
    artifact = Path(path)
    # Validate the complete artifact before any mutation.  This keeps signing
    # and signoff from rewriting a stale/tampered policy into an apparently
    # valid state.
    load_policy(artifact)
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    return artifact, doc


def _write(artifact: Path, doc: dict) -> None:
    _validate_structural(doc)
    atomic_write_json(artifact, doc)


def cmd_sign(args: argparse.Namespace) -> int:
    signing_key = os.environ.get("POLICY_SIGNING_KEY", "")
    if not signing_key:
        print("FAIL: POLICY_SIGNING_KEY must be set in the environment (never commit it)", file=sys.stderr)
        return 3
    artifact, doc = _load(args.policy)
    value = sign_policy_payload(doc["policy_version"], doc["content"], signing_key)
    key_id = f"hmac-{value[:12]}"
    doc["signatures"] = [{
        "alg": "HMAC-SHA256",
        "key_id": key_id,
        "value": value,
        "signed_by": args.signed_by,
        "date": date.today().isoformat(),
        "covers": {"policy_version": doc["policy_version"], "content_sha256": doc["content_sha256"]},
    }]
    _write(artifact, doc)
    print(f"SIGNED: policy v{doc['policy_version']} signed by {args.signed_by} (key_id={key_id})")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    signing_key = os.environ.get("POLICY_SIGNING_KEY", "")
    if not signing_key:
        print("FAIL: POLICY_SIGNING_KEY must be set in the environment", file=sys.stderr)
        return 3
    policy = load_policy(args.policy)
    if verify_signatures(policy, signing_key):
        print(f"VERIFIED: signature valid for policy v{policy.policy_version} sha256:{policy.content_sha256[:12]}")
        return 0
    print("FAIL: no signature matches the current key/version/fingerprint", file=sys.stderr)
    return 3


def _write_signoff(args: argparse.Namespace, status: str) -> int:
    artifact, doc = _load(args.policy)
    sha = canonical_sha256(doc["content"])
    if doc.get("content_sha256") != sha:
        print("FAIL: content fingerprint mismatch — artifact was tampered after signing", file=sys.stderr)
        return 2
    doc["clinical_signoff"] = {
        "status": status,
        "reviewer": args.reviewer,
        "credentials_ref": args.credentials,
        "date": date.today().isoformat(),
        "covers_sha256": sha,
        "scope": args.scope or ("急救词表、剂量话术与免责文本的临床适当性" if status == "approved" else ""),
        "notes": getattr(args, "notes", "") or "",
    }
    _write(artifact, doc)
    print(f"SIGNOFF[{status}]: policy v{doc['policy_version']} reviewer={args.reviewer} covers sha256:{sha[:12]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="policy artifact signing & clinical signoff")
    sub = parser.add_subparsers(dest="command", required=True)

    # 每个子命令各自接受 --policy(允许 `sign --policy PATH` 的自然顺序)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--policy", default=config.safety_policy_path)

    sign = sub.add_parser("sign", parents=[common], help="append/replace the maintainer HMAC signature")
    sign.add_argument("--signed-by", default="维护者")
    sign.set_defaults(func=cmd_sign)

    verify = sub.add_parser("verify", parents=[common], help="verify the maintainer signature")
    verify.set_defaults(func=cmd_verify)

    start = sub.add_parser("start-signoff", parents=[common], help="record a pending clinical review")
    start.add_argument("--reviewer", required=True)
    start.add_argument("--credentials", required=True)
    start.add_argument("--scope", default="")
    start.set_defaults(func=lambda a: _write_signoff(a, "pending"))

    approve = sub.add_parser("approve-signoff", parents=[common], help="record an approved clinical signoff")
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--credentials", required=True)
    approve.add_argument("--scope", default="")
    approve.add_argument("--notes", default="")
    approve.set_defaults(func=lambda a: _write_signoff(a, "approved"))

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (PolicyArtifactError, PolicyReviewError, json.JSONDecodeError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
