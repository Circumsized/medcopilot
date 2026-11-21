"""Project-local pytest bootstrap for ``medcopilot/tests/``.

Ensures that running ``pytest`` directly inside ``medcopilot/`` (without
loading the workspace root ``conftest.py``) still gets:
1. ``PYTHONPATH`` / ``sys.path`` containing the parent directory so
   ``import medcopilot.*`` resolves without an editable install;
2. Explicit ``PROJECT_ENV=development``, ``MED_RATE_LIMIT_ENABLED=0``, and
   ``RERANK_ENABLED=0`` defaults;
3. Per-process isolated ``LOG_FILE`` and ``AUDIT_LOG_PATH`` in the system temp
   directory (never ``logs/app.log`` or ``logs/audit.jsonl`` inside the repo).
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_PARENT = Path(__file__).resolve().parents[2]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

os.environ.setdefault("PROJECT_ENV", "development")
os.environ.setdefault("MED_RATE_LIMIT_ENABLED", "0")
os.environ.setdefault("RERANK_ENABLED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

if "LOG_FILE" not in os.environ or "AUDIT_LOG_PATH" not in os.environ:
    _TEST_TMP = Path(tempfile.mkdtemp(prefix=f"med-rag-tests-{os.getpid()}-"))
    atexit.register(shutil.rmtree, _TEST_TMP, ignore_errors=True)
    os.environ.setdefault("LOG_FILE", str(_TEST_TMP / "app.log"))
    os.environ.setdefault("AUDIT_LOG_PATH", str(_TEST_TMP / "audit.jsonl"))

if subprocess.Popen.__name__ != "_IsolatingPopen":
    _MANAGED_CHILD_PATHS = {
        "LOG_FILE": os.environ["LOG_FILE"],
        "AUDIT_LOG_PATH": os.environ["AUDIT_LOG_PATH"],
    }
    _CHILD_BASE = Path(os.environ["LOG_FILE"]).parent
    _child_spawn_seq = 0
    _RealPopen = subprocess.Popen

    def _isolated_child_env(env: dict | None) -> dict | None:
        global _child_spawn_seq
        base = dict(os.environ if env is None else env)
        replacements: dict[str, str] = {}
        for key, managed in _MANAGED_CHILD_PATHS.items():
            if managed and base.get(key) == managed:
                _child_spawn_seq += 1
                target = _CHILD_BASE / f"child-{os.getpid()}-{_child_spawn_seq}" / Path(managed).name
                target.parent.mkdir(parents=True, exist_ok=True)
                replacements[key] = str(target)
        if not replacements:
            return env
        base.update(replacements)
        return base

    class _IsolatingPopen(_RealPopen):
        def __init__(self, *args, **kwargs):
            kwargs["env"] = _isolated_child_env(kwargs.get("env"))
            super().__init__(*args, **kwargs)

    subprocess.Popen = _IsolatingPopen
