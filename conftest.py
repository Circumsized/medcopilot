import os
import subprocess
import sys
import tempfile
from pathlib import Path

# W1.3 default-deny: the test suite runs in an explicit dev environment so the
# historical local unauthenticated flow keeps working; an empty or misspelled
# PROJECT_ENV now fails closed in production code (see api/env_policy.py).
os.environ.setdefault("PROJECT_ENV", "development")

# W2.3: the per-identity API quotas are a production control. Tests (and the
# subprocess servers they spawn, which inherit this env) hammer one client
# address far beyond 20 req/min, so they run with the limiter off; the limiter
# contract has its own tests that switch it on explicitly.
os.environ.setdefault("MED_RATE_LIMIT_ENABLED", "0")

# Reranker model loading is an optional integration concern. Keep ordinary tests
# deterministic and avoid native transformer crashes; reranker-specific tests
# replace CrossEncoder with a fake when they need the enabled path.
os.environ.setdefault("RERANK_ENABLED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Every test process writes its own log file instead of the repo's shared
# logs/app.log. Two processes appending to that one file make loguru's rotation
# (rename + zip) fail on Windows, and the resulting error text lands in whatever
# subprocess output a gate is parsing — which made a release-gate assertion fail
# based on how much had been logged earlier in the day.
#
# Both sinks live in one per-process temp directory (never the repo's prod
# logs/ tree): LOG_FILE above, and AUDIT_LOG_PATH here. The audit sink is a
# hash-chained append log whose AppConfig default (logs/audit.jsonl) points
# straight into that prod tree, and two processes appending to one chain
# interleave records and corrupt it; a shared log file additionally breaks
# loguru's rename-based rotation on Windows.
_TEST_TMP = Path(tempfile.mkdtemp(prefix=f"med-rag-tests-{os.getpid()}-"))
os.environ.setdefault("LOG_FILE", str(_TEST_TMP / "app.log"))
os.environ.setdefault("AUDIT_LOG_PATH", str(_TEST_TMP / "audit.jsonl"))

# Parent-child isolation: the two values above are inherited verbatim by every
# subprocess a test spawns (the uvicorn server under load_check, the release-pack
# gate subprocesses), which would put parent and child back on one shared file.
# _IsolatingPopen re-points an *inherited* value to a per-spawn path while
# leaving explicit test choices (LOG_FILE="" to disable the sink, a tmp_path
# audit target) exactly as written — only values identical to what this conftest
# set are rewritten, so no test-visible environment silently changes.
_MANAGED_CHILD_PATHS = {
    "LOG_FILE": os.environ["LOG_FILE"],
    "AUDIT_LOG_PATH": os.environ["AUDIT_LOG_PATH"],
}
_child_spawn_seq = 0


def _isolated_child_env(env: dict | None) -> dict | None:
    global _child_spawn_seq
    base = dict(os.environ if env is None else env)
    replacements: dict[str, str] = {}
    for key, managed in _MANAGED_CHILD_PATHS.items():
        if base.get(key) == managed:
            _child_spawn_seq += 1
            target = _TEST_TMP / f"child-{os.getpid()}-{_child_spawn_seq}" / Path(managed).name
            target.parent.mkdir(parents=True, exist_ok=True)
            replacements[key] = str(target)
    if not replacements:
        return env
    base.update(replacements)
    return base


_RealPopen = subprocess.Popen


class _IsolatingPopen(_RealPopen):
    def __init__(self, *args, **kwargs):
        kwargs["env"] = _isolated_child_env(kwargs.get("env"))
        super().__init__(*args, **kwargs)


subprocess.Popen = _IsolatingPopen

# Ensure project root is importable without requiring editable install
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
