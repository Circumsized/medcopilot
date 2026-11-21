"""Structured log and immutable supply-chain contract checks."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_json_log_format_has_request_id_and_redacts_query_secrets():
    command = [sys.executable, "-c", (
        "from medcopilot.common.logger import logger; "
        "logger.bind(request_id='req-123').info('provider ?access_token=secret-value api_key=secret-other')"
    )]
    env = {**os.environ, "LOG_FORMAT": "json", "LOG_FILE": "", "LOG_LEVEL": "INFO", "PYTHONPATH": "."}
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    line = result.stdout.strip()
    data = json.loads(line)
    assert data["record"]["extra"]["request_id"] == "req-123"
    assert "secret-value" not in line and "secret-other" not in line


def test_all_actions_are_pinned_to_verified_commit_shas():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    actions = re.findall(r"uses:\s*([^\s#]+)", workflow)
    assert actions
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", action) for action in actions)


def test_docker_uses_hashed_dependencies_and_pinned_base_without_online_upgrade():
    docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"FROM python:3\.12-slim@sha256:[0-9a-f]{64}", docker)
    assert "--require-hashes" in docker
    assert "pip install --upgrade" not in docker
    assert "/health/live" in docker
    assert '"--workers", "1"' in docker or "entrypoint" in docker


def test_runtime_lock_contains_pinned_hashed_packages_no_editable_project():
    text = (ROOT / "requirements.lock.txt").read_text(encoding="utf-8")
    assert "--hash=sha256:" in text
    assert "-e ." not in text
    entries = [line for line in text.splitlines() if re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*", line)]
    assert entries and all("==" in entry for entry in entries)
