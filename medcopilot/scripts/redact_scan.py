"""Fail-closed scan for secrets and PHI-like fields in release evidence."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REDACTED_VALUE_RE = re.compile(r"(?i)^(?:(?:bearer|basic)\s+)?\*{3,}$")

PATTERNS = (
    re.compile(
        r'(?i)(?:\\"|["\'])?(api[_-]?key|access[_-]?token|client[_-]?secret|authorization)(?:\\"|["\'])?'
        r'\s*[:=]\s*(?:\\"|["\'])?(?P<value>[^,}\]"\\\'\n]+)'
    ),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r'(?i)(?:\\"|["\'])?(患者主诉|身份证|手机号|phone|email|date_of_birth)(?:\\"|["\'])?\s*[:=]'),
)


def _has_unredacted_match(pattern: re.Pattern[str], text: str) -> bool:
    for match in pattern.finditer(text):
        if "value" in match.groupdict():
            cleaned = match.group("value").strip()
            if not cleaned or _REDACTED_VALUE_RE.fullmatch(cleaned):
                continue
        return True
    return False


def scan_path(path: Path) -> list[str]:
    findings: list[str] = []
    if not path.exists():
        # Fail closed: a scan target that does not exist is a wiring bug (a
        # typo'd CI path, a cleaned workspace, a renamed artifact), not a clean
        # tree. Iterating nothing and reporting PASS is exactly how an unscanned
        # release gets shipped.
        return [f"{path}: scan target does not exist (fail-closed)"]
    for candidate in sorted(path.rglob("*") if path.is_dir() else [path]):
        if not candidate.is_file() or candidate.suffix not in {".json", ".jsonl", ".txt", ".log"}:
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append(f"{candidate}: unreadable: {exc}")
            continue
        active_patterns = PATTERNS[:2] if candidate.name == "sbom.json" else PATTERNS
        for pattern in active_patterns:
            if _has_unredacted_match(pattern, text):
                findings.append(f"{candidate}: matched {pattern.pattern}")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="scan release evidence for secrets or PHI-like fields")
    parser.add_argument("path", nargs="?", default="outputs")
    parser.add_argument(
        "--output",
        default=None,
        help="write a summary JSON (counts only, never finding text) for release_pack to consume",
    )
    args = parser.parse_args(argv)
    findings = scan_path(Path(args.path))
    if args.output:
        summary = {"report_schema_version": 1, "all_passed": not findings, "findings_count": len(findings)}
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if findings:
        print("REDACT SCAN: FAIL", file=sys.stderr)
        print("\n".join(findings), file=sys.stderr)
        return 1
    print("REDACT SCAN: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
