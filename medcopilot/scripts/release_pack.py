"""发布证据包 CLI（Slice 7）。

真实子进程执行各验收门并把结果聚合为单一 JSON 证据包——只聚合、不修饰：
- 必须门（全量非 slow 测试 / 配置镜像一致性 / 金标行为门 / 策略门禁 /
  急救红旗验收集 emergency_eval）任一失败 ⇒ 整体 FAIL。
- ``rag_quality``（eval_rag）在 mock 下**按设计诚实 FAIL**、``clinical_signoff``
  在无资质签核时按设计 FAIL：记录为 ``expected_fail``，不影响整体通过——
  不调阈值、不用免责文本凑 PASS。
- 退出码分类：0 ⇒ pass；退出码命中该门的 ``expected_exit_codes`` ⇒
  必须门 fail / 非必须门 expected_status；**其余退出码（含超时）⇒
  ``crash``**——崩溃或卡死的工具不产出证据，无论必须与否一律阻断。

用法::

    python scripts/release_pack.py [--only policy_gate,golden_gate] [--output outputs/release_pack.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medcopilot.common.fsutil import atomic_write_json, git_metadata
from medcopilot.llm.unified_llm import UnifiedLLM

PACK_SCHEMA_VERSION = 2
PY = sys.executable
REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_TIMEOUT_SECONDS = 1800


def _redact_tail(lines: list[str]) -> list[str]:
    """Scrub credentials from gate output tails via the existing ``UnifiedLLM._redact``.

    The pack embeds gate stdout/stderr tails verbatim, and the pack is uploaded
    as a CI artifact — a provider key echoed in one error line would otherwise
    be copied straight into the evidence bundle. Reuse the single scrubbing
    implementation the rest of the codebase already relies on (CWE-209/532)
    instead of maintaining a second, drifting regex here.
    """
    return [UnifiedLLM._redact(line) for line in lines]


def _report_gate(
    name: str,
    command: list[str],
    *,
    required: bool = True,
    expected_exit_codes: list[int] | tuple[int, ...] = (0,),
) -> dict[str, Any]:
    return {
        "command": command,
        "required": required,
        "expected_status": "pass",
        "expected_exit_codes": list(expected_exit_codes),
        "env": {"MOCK_LLM": "1"},
    }

# name -> spec。``expected_exit_codes`` 显式列出该门"可解释"的已知退出码。
# 省略该键（None）时任何非零都是普通失败；显式给出集合后启用崩溃可分辨化——
# 命中集合的非零码按 required/expected_status 归类，集合外（含未列出的意外
# 退出码）⇒ "crash"。因此文档规定 exit 1 为普通失败的门必须写 [0, 1]，
# 否则一次真实的失败会被误报成工具崩溃（recovery_drill / load_check /
# redact_scan 失败时仍写出报告，exit 1 是可解释的失败而非崩溃）。
GATES: dict[str, dict[str, Any]] = {
    "unit_tests": {
        "command": [PY, "-m", "pytest", "medcopilot/tests/", "-q", "-m", "not slow"],
        "required": True,
        "expected_status": "pass",
        "env": {"MOCK_LLM": "1"},
    },
    "config_mirrors": {
        "command": [PY, "-m", "pytest", "medcopilot/tests/test_config_mirrors.py", "-q"],
        "required": True,
        "expected_status": "pass",
        "env": {"MOCK_LLM": "1"},
    },
    "golden_gate": {
        "command": [PY, "medcopilot/scripts/eval_golden.py"],
        "required": True,
        "expected_status": "pass",
        "env": {"MOCK_LLM": "1"},
    },
    "policy_gate": {
        "command": [PY, "medcopilot/scripts/policy_check.py"],
        "required": True,
        "expected_status": "pass",
        "env": {"MOCK_LLM": "1"},
    },
    # 诚实质量声明：mock 下 eval_rag FAIL 是设计内行为（安全弃答不算达标）。
    # eval_rag 约定：0=达标 / 1=设计内诚实 FAIL；其余（如 2）⇒ 崩溃。
    "rag_quality": {
        "command": [PY, "medcopilot/scripts/eval_rag.py"],
        "required": False,
        "expected_exit_codes": [0, 1],
        "expected_status": "expected_fail",
        "env": {"MOCK_LLM": "1"},
    },
    # W1.1 急救红旗验收集门禁（红标召回=1.0 / 误报≤0.05 / 降级准确=1.0）。
    "emergency_eval": {
        "command": [PY, "-m", "medcopilot.scripts.emergency_eval"],
        "required": True,
        "expected_status": "pass",
        "env": {"MOCK_LLM": "1"},
    },
    "recovery_drill": _report_gate(
        "recovery_drill", [PY, "medcopilot/scripts/recovery_drill.py", "--output", "outputs/recovery_drill.json"],
        # Documented outcome: 0 = every scenario passed, 1 = a scenario failed
        # (the report is still written). A real failure, not a tool crash.
        expected_exit_codes=(0, 1),
    ),
    "load_check": _report_gate(
        "load_check",
        [PY, "medcopilot/scripts/load_check.py", "--concurrency", "4",
         "--requests-per-worker", "2", "--output", "outputs/load_report.json"],
        # Same load profile as the CI "Load check" step (4x2): the evidence gate and
        # the CI step must measure the same thing, not two different profiles.
        # Documented outcome: 0 = no errors, 1 = errors/not-ready (report written).
        expected_exit_codes=(0, 1),
        required=True,
    ),
    "redact_scan": _report_gate(
        "redact_scan",
        [PY, "medcopilot/scripts/redact_scan.py", "outputs", "--output", "outputs/redact_scan.json"],
        # Documented outcome: 0 = clean, 1 = findings (summary written). A finding
        # is a failure of the scanned evidence, not a broken scanner.
        expected_exit_codes=(0, 1),
        required=True,
    ),
    # W6.2 maintainer signature and clinical signoff are independently visible.
    "policy_signature": {
        "command": [PY, "medcopilot/scripts/policy_check.py", "--require-signature"],
        "required": False,
        "expected_exit_codes": [0, 3],
        "expected_status": "expected_fail",
        "env": {"MOCK_LLM": "1"},
    },
    # （不阻断,但不可伪装）。资质评审人签核后此门自动转为 PASS;发布方应将其
    # 升级为 required=True 以阻断未签核发布。
    "clinical_signoff": {
        "command": [PY, "medcopilot/scripts/policy_check.py", "--require-signoff"],
        "required": False,
        "expected_exit_codes": [4],
        "expected_status": "expected_fail",
        "env": {"MOCK_LLM": "1"},
    },
}


def aggregate_gates(gates: list[dict[str, Any]], *, partial: bool = False, profile: str | None = None) -> str:
    """Return a release decision, never treating a subset as a formal pass."""
    for gate in gates:
        status = gate.get("status")
        if status not in {"pass", "fail", "expected_fail", "crash"}:
            raise ValueError(f"unknown gate status: {status!r}")
        if status == "crash" or (gate.get("required") and status != "pass"):
            return "fail"
    if partial:
        return "incomplete"
    if profile == "opensource" and any(
        gate.get("name") == "clinical_signoff" and gate.get("status") != "pass"
        for gate in gates
    ):
        return "engineering_pass_clinical_blocked"
    return "pass"


def gate_status(
    returncode: int,
    required: bool,
    expected_status: str,
    expected_exit_codes: list[int] | tuple[int, ...] | None = None,
) -> str:
    """把门禁子进程退出码映射为证据状态。

    必须门的失败是硬失败，绝不被 expected_status 洗白成 pass；只有非必须门
    允许"按设计失败"（记录为 expected_fail，不阻塞整体结论）。

    0 恒为 pass。``expected_exit_codes=None``（缺省）保持既有语义：任何非零
    都是普通失败/expected_status。显式给定集合后启用崩溃可分辨化：命中集合
    的非零码按必须性归类，**其余退出码是工具崩溃（未产出可信证据）⇒
    "crash"**，无论必须与否。
    """
    if returncode == 0:
        return "pass"
    if expected_exit_codes is not None and returncode not in expected_exit_codes:
        return "crash"
    return "fail" if required else expected_status


def _run_gate(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = {**GATES[name], **(overrides or {})}
    env = {**os.environ, **spec["env"]}
    started = time.perf_counter()
    command = spec["command"]
    try:
        completed = subprocess.run(
            command, cwd=REPO_ROOT,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=GATE_TIMEOUT_SECONDS, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        # A timed-out gate must still leave a pack behind: preserve whatever
        # partial output exists and classify it as crash (it blocked, so it
        # produced no trustworthy evidence) — never traceback the whole run.
        # `output` is the canonical attribute carrying the captured stdout
        # (`stdout` is only a compatibility alias for it), so it is read first
        # and the alias kept as a fallback.
        partial = getattr(exc, "output", None) or getattr(exc, "stdout", None) or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        partial_err = getattr(exc, "stderr", None) or ""
        if isinstance(partial_err, bytes):
            partial_err = partial_err.decode("utf-8", errors="replace")
        return {
            "name": name,
            "command": command[:3] + ["..."] if len(command) > 3 else command,
            "exit_code": None,
            "status": "crash",
            "required": spec["required"],
            "duration_seconds": round(time.perf_counter() - started, 3),
            "error": f"timed out after {GATE_TIMEOUT_SECONDS}s",
            "stdout_tail": _redact_tail(partial.strip().splitlines()[-20:]) if partial.strip() else [],
            "stderr_tail": _redact_tail(partial_err.strip().splitlines()[-20:]) if partial_err.strip() else [],
            "profile": spec.get("profile", "opensource"),
        }
    duration = time.perf_counter() - started
    status = gate_status(
        completed.returncode, spec["required"], spec["expected_status"], spec.get("expected_exit_codes")
    )
    out_text = completed.stdout or ""
    err_text = completed.stderr or ""
    stdout_tail = (out_text + "\n" + err_text).strip().splitlines()
    return {
        "name": name,
        "command": command[:3] + ["..."] if len(command) > 3 else command,
        "exit_code": completed.returncode,
        "status": status,
        "required": spec["required"],
        "duration_seconds": round(duration, 3),
        "stdout_tail": _redact_tail(stdout_tail[-20:]),
        "stderr_tail": _redact_tail(err_text.strip().splitlines()[-20:]) if err_text.strip() else [],
        "profile": spec.get("profile", "opensource"),
    }


def _ops_evidence(report_path: Path) -> dict[str, Any]:
    """Return a bounded, machine-readable summary; missing is not silently omitted."""
    if not report_path.is_file():
        return {"status": "missing", "path": str(report_path)}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "invalid", "path": str(report_path), "error": str(exc)}
    return {
        "status": "available",
        "path": str(report_path),
        "schema": report.get("report_schema_version", report.get("load_schema_version")),
        "all_passed": report.get("all_passed"),
        "ready": report.get("ready"),
        "errors": report.get("errors"),
        "total_requests": report.get("total_requests"),
        "latency_p95_seconds": report.get("latency_p95_seconds"),
    }


def _evidence_green(name: str, evidence: dict[str, Any]) -> bool:
    """A report that exists but says "failed" must not satisfy a production gate."""
    if name == "load_check":
        return evidence.get("ready") is True and evidence.get("errors") == 0
    return evidence.get("all_passed") is True


def build_pack(
    selected: list[str], *, profile: str = "opensource", evidence_paths: dict[str, Path] | None = None
) -> dict[str, Any]:
    gates = []
    for name in selected:
        overrides = {"profile": profile}
        if profile == "production" and name in {"clinical_signoff", "policy_signature"}:
            # Exit 3 (no valid signature) / 4 (no approved signoff) are the documented
            # "unsigned" outcomes: a required gate that fails, not a tool crash.
            unsigned_exit = 3 if name == "policy_signature" else 4
            overrides.update(
                {"required": True, "expected_status": "pass", "expected_exit_codes": [0, unsigned_exit]}
            )
        gates.append(_run_gate(name, overrides))
    partial = set(selected) != set(GATES)
    production_required = {
        "unit_tests", "config_mirrors", "golden_gate", "policy_gate", "emergency_eval",
        "recovery_drill", "load_check", "redact_scan", "policy_signature", "clinical_signoff",
    }
    profile_blockers: list[str] = []
    if profile == "production":
        missing_production = production_required - set(selected)
        if missing_production:
            profile_blockers.append("missing_required_gates:" + ",".join(sorted(missing_production)))
        if (os.getenv("PROJECT_ENV") or "").strip().lower() not in {"production", "prod"}:
            profile_blockers.append("production_environment_required")
        if (os.getenv("MOCK_LLM") or "").strip().lower() in {"1", "true", "yes", "on"}:
            profile_blockers.append("mock_llm_enabled")
    ops_paths = evidence_paths if evidence_paths is not None else {
        "recovery_drill": REPO_ROOT / "outputs" / "recovery_drill.json",
        "load_check": REPO_ROOT / "outputs" / "load_report.json",
        "redact_scan": REPO_ROOT / "outputs" / "redact_scan.json",
    }
    ops_evidence = {
        name: _ops_evidence(Path(ops_paths[name])) if name in ops_paths else {"status": "missing"}
        for name in ("recovery_drill", "load_check", "redact_scan")
    }
    if profile == "production":
        for name in ("recovery_drill", "load_check", "redact_scan"):
            evidence = ops_evidence.get(name, {})
            if evidence.get("status") != "available":
                profile_blockers.append(f"missing_{name}_evidence")
            elif not _evidence_green(name, evidence):
                profile_blockers.append(f"failed_{name}_evidence")
    overall = "fail" if profile_blockers else aggregate_gates(gates, partial=partial, profile=profile)
    return {
        "pack_schema_version": PACK_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_metadata(),
        "gates": gates,
        "partial": partial,
        "incomplete": partial,
        "profile": profile,
        "profile_blockers": profile_blockers,
        "overall": overall,
        "ops_evidence": ops_evidence,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="aggregate release evidence gates into one JSON pack")
    parser.add_argument(
        "--only",
        default=",".join(GATES),
        help=f"comma-separated gate subset (default: all of {', '.join(GATES)})",
    )
    parser.add_argument("--profile", choices=("production", "opensource"), default="opensource")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "release_pack.json"),
        help="evidence pack destination (default: <repo>/outputs/release_pack.json)",
    )
    args = parser.parse_args(argv)

    selected = [part.strip() for part in args.only.split(",") if part.strip()]
    unknown = set(selected) - set(GATES)
    if unknown:
        print(f"FAIL: unknown gates: {sorted(unknown)}; available: {sorted(GATES)}", file=sys.stderr)
        return 2

    pack = build_pack(selected, profile=args.profile)
    raw_output = Path(args.output).expanduser()
    # Relative destinations are repo-relative just like gate cwd and GPU
    # evidence; absolute paths remain an explicit caller choice (tests and
    # external artifact collectors may use tmp directories).
    out = raw_output if raw_output.is_absolute() else REPO_ROOT / raw_output
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a half-written evidence pack is an unreadable evidence pack.
    atomic_write_json(out, pack)

    print(f"RELEASE PACK: overall={pack['overall']} gates={selected} report={out}")
    for gate in pack["gates"]:
        print(f"  [{gate['status']}] {gate.get('name', 'unknown')} exit={gate['exit_code']} ({gate['duration_seconds']}s)")
    return 0 if pack["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
