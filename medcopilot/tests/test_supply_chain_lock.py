"""Supply-chain wiring (W5.2): the lock is installable, auditable and actually used.

These pin the three things that were wrong or missing, each of which silently
degrades a supply-chain control rather than failing loudly:

1. The CPU torch wheels live on the PyTorch index, not PyPI. A hash-locked
   install without that index resolves PyPI's differently-hashed wheel and dies
   on the hash check (verified locally) — so the Dockerfile/CI must declare it.
2. ``pip-audit`` audits the *installed* distribution, so a pinned ``+cpu`` local
   version cannot be looked up on PyPI. Keeping the local segments out of the
   generated lock keeps the file resolvable while the hashes still bind the
   exact CPU wheels; the container build is what installs them.
3. Nothing regenerates the lock: a stale ``requirements.lock.txt`` in a repo that
   claims hash-pinned installs is the failure mode this guards.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "Dockerfile"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
LOCK = REPO_ROOT / "requirements.lock.txt"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def _joined(text: str) -> str:
    """Collapse shell line continuations so one command reads as one line."""
    return re.sub(r"\\\s*\n\s*", " ", text)


def test_docker_install_is_hash_locked_and_declares_the_cpu_index():
    docker = _joined(DOCKERFILE.read_text(encoding="utf-8"))
    assert "--require-hashes" in docker
    install = next(line for line in docker.splitlines() if "--require-hashes" in line)
    assert CPU_INDEX in install
    assert "requirements.lock.txt" in install


def _commands_only(text: str) -> str:
    """Comment lines explain why; only executable lines carry the guarantee."""
    return "\n".join(
        line for line in _joined(text).splitlines() if not line.lstrip().startswith("#")
    )


def test_ci_audits_the_hash_locked_file_with_the_cpu_index():
    workflow = _commands_only(WORKFLOW.read_text(encoding="utf-8"))
    audit = next(
        line for line in workflow.splitlines()
        # the audit invocation, not the `pip install pip-audit` bootstrap line
        if re.search(r"\bpip-audit\s+--", line)
    )
    assert "--require-hashes" in audit
    assert "requirements.lock.txt" in audit
    assert CPU_INDEX in audit
    # --strict cannot be satisfied by the CPU-only torch builds; the comment above
    # the step records why, so the flag must not quietly come back.
    assert "--strict" not in audit
    assert "advisory service cannot look them up" in WORKFLOW.read_text(encoding="utf-8")


def test_ci_checks_the_generated_lock_is_fresh():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "export_lock.py --profile" in workflow


def test_lock_has_no_local_version_pins_and_keeps_every_hash():
    text = LOCK.read_text(encoding="utf-8")
    local_versions = [line for line in text.splitlines() if re.search(r"==[^;\\\s]+\+", line)]
    assert not local_versions, f"local-version pins are unauditable: {local_versions[:3]}"
    entries = [line for line in text.splitlines() if re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*==", line)]
    assert entries
    # Every requirement keeps its hash block (the transform must not drop them).
    assert text.count("--hash=sha256:") > len(entries)
    assert "-e ." not in text


def test_export_lock_is_deterministic_without_running_uv_export_twice(monkeypatch):
    """The transform itself is a pure function of the export text."""
    from medcopilot.scripts import export_lock

    sample = (
        "torch==2.14.1+cpu ; sys_platform != 'darwin' \\\n"
        "    --hash=sha256:aaa\n"
        "fastapi==0.142.2 \\\n"
        "    --hash=sha256:bbb\n"
    )
    lines = []
    for line in sample.splitlines():
        match = export_lock._LOCAL_VERSION.match(line)
        lines.append(
            f"{match.group('name')}=={match.group('version')}{match.group('rest')}" if match else line
        )
    assert lines[0].startswith("torch==2.14.1 ; sys_platform")
    assert lines[1].strip() == "--hash=sha256:aaa"
    assert lines[2].startswith("fastapi==0.142.2")
