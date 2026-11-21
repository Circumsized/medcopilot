"""Optional Locust load-test POC for the running Med-RAG API.

This file is intentionally outside the default test/CI path.  Locust is an
optional release-time dependency; when it is not installed the module remains
importable and reports ``not-configured`` from its CLI instead of pretending a
load test ran.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from locust import HttpUser, between, task
    LOCUST_CONFIGURED = True
except ImportError:  # pragma: no cover - exercised in environments without Locust
    HttpUser = object  # type: ignore[assignment,misc]
    LOCUST_CONFIGURED = False

    def between(*_args: Any, **_kwargs: Any) -> Any:
        return None

    def task(*_args: Any, **_kwargs: Any):
        def decorate(function):
            return function

        return decorate


class ConsultationUser(HttpUser):
    """Small, safe read-only consultation workload for manual release checks."""

    wait_time = between(0.2, 1.0)

    @task
    def consultation(self) -> None:
        self.client.post(
            "/consultations",
            json={"query": "负载检查：请说明服务当前可提供的帮助"},
            name="POST /consultations",
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="optional Locust POC; run with 'locust -f medcopilot/loadtests/locustfile.py'"
    )
    parser.add_argument("--check", action="store_true", help="only check whether Locust is installed")
    args = parser.parse_args(argv)
    configured = LOCUST_CONFIGURED
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "locust_consultation_poc",
        "status": "configured" if configured else "not-configured",
        "locust_file": str(Path(__file__).resolve().relative_to(_repo_root())),
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if args.check or not configured:
        return 0 if configured else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
