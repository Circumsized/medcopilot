"""Container launcher for the Med-RAG API (W5.5/W5.7 deploy contract).

Docker secret channel: a dotenv-syntax file (docker/compose secret, default
``/run/secrets/med_env``). The shell wrapper (``deploy/docker-entrypoint.sh``)
never ``source``s that file — sourcing executes its content as shell code.
Instead this module parses it with ``python-dotenv`` (pure data parsing with
``interpolate=False``) and copies a fixed allowlist of AppConfig aliases into
the process environment, then ``exec``s uvicorn. The application — including
its ``AppConfig`` import at startup — therefore only ever runs with the merged
environment, i.e. secrets are parsed and filtered BEFORE any app import.

Contract enforced here (pinned by ``tests/test_deploy_contract.py``):

* no shell ``source``/``eval`` of the secret file;
* values stay literal — ``$(...)`` / ``${VAR}`` interpolation is disabled;
* only :data:`ALLOWLIST_ENV_KEYS` (exactly the AppConfig aliases) is exported,
  so interpreter hooks such as ``PYTHONPATH`` / ``LD_PRELOAD`` can never enter
  the process through the secret file and are scrubbed when inherited;
* ``PROJECT_ENV`` must classify as ``production`` (fail closed);
* secret values are never printed — diagnostics carry names and counts only.

systemd deployments do NOT use this launcher: systemd parses
``EnvironmentFile=`` itself (KEY=VALUE, never executed), so
``deploy/med-rag.service`` execs ``uvicorn`` directly. Both paths converge on
the same single-worker command documented in ``docs/PRODUCTION_DEPLOY.md``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

DEFAULT_SECRET_FILE = "/run/secrets/med_env"
APP_MODULE = "medcopilot.api.server:app"
BIND_HOST = "0.0.0.0"
BIND_PORT = 8000
# Hard invariant (docs/RELEASE_CHECKLIST.md §2): the session store and the
# authoritative corpus store are process-local; more than one worker forks the
# serving state. Never raise this here or in the Dockerfile CMD.
WORKERS = 1
# Trust-boundary sanity cap: a dotenv file that grows past 1 MiB is a
# misconfiguration or an oversized blob; refuse it instead of parsing.
MAX_SECRET_BYTES = 1024 * 1024
# Never exported from a secret file and scrubbed when inherited: they
# re-route/interpose the interpreter itself instead of configuring the app.
INTERPRETER_HOOKS = ("PYTHONPATH", "LD_PRELOAD")

# Exactly the AppConfig field aliases (medcopilot.common.config). The
# allowlist is intentionally hardcoded — importing AppConfig here would
# instantiate settings against the pre-secret environment. tests/
# test_deploy_contract.py asserts set-equality, so a new config key fails CI
# until this allowlist is extended on purpose.
ALLOWLIST_ENV_KEYS = frozenset(
    {
        "ADMIN_API_KEY",
        "ALLOW_MOCK_IN_PRODUCTION",
        "APP_GIT_SHA",
        "APP_VERSION",
        "AUDIT_FSYNC",
        "AUDIT_LOG_PATH",
        "AUDIT_NTP_SERVER",
        "AUDIT_RETENTION_DAYS",
        "CHROMA_PERSIST_DIRECTORY",
        "CORS_ALLOW_ALL",
        "CORS_ORIGINS",
        "DOSAGE_GATE_MODE",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_CACHE_DIR",
        "EMBEDDING_DIM",
        "EMBEDDING_MODEL",
        "EMBEDDING_NORMALIZE",
        # W4 评测通道的 API 凭据:进允许清单只为与 AppConfig 别名保持精确相等
        # (契约测试强制);生产默认不启用该通道(见 rag/embedding_api.py 边界)。
        "EMBEDDING_API_BASE",
        "EMBEDDING_API_KEY",
        "EMBEDDING_API_TIMEOUT",
        "EMBEDDING_PROVIDER",
        "ENABLE_CHROMA_DENSE",
        "EVIDENCE_FLOOR",
        "EVIDENCE_TOP_N",
        "GROUNDEDNESS_THRESHOLD",
        "INGEST_MAX_LINE_CHARS",
        "INGEST_MIN_RAW_CHARS",
        "INGEST_VALIDATION_ENABLED",
        "LLM_API_BASE",
        "LLM_API_KEY",
        "LLM_BACKOFF_BASE",
        "LLM_BACKOFF_CAP",
        "LLM_MAX_ATTEMPTS",
        "LLM_MAX_TOKENS",
        "LLM_MODEL",
        "LLM_PROVIDER",
        "LLM_MODE",
        "LLM_RETRY_DEADLINE",
        "LLM_TEMPERATURE",
        "LLM_TIMEOUT",
        "LOG_FILE",
        "LOG_FORMAT",
        "LOG_LEVEL",
        "MAX_DRUG_LABEL_CHARS",
        "MED_API_BASE_URL",
        "MED_API_TIMEOUT_SECONDS",
        "MED_CORPUS_PATH",
        "MED_LLM_CHUNK_TOKEN_CAP",
        "MED_LLM_CONCURRENCY",
        "MED_LLM_EVIDENCE_TOKEN_BUDGET",
        "MED_LLM_PROMPT_TOKEN_BUDGET",
        "MED_OCR_MAX_LINE_CHARS",
        "MED_OCR_MAX_LINES",
        "MED_RATE_LIMIT_CONSULT_PER_MINUTE",
        "MED_RATE_LIMIT_ENABLED",
        "MED_RATE_LIMIT_IMAGE_PER_MINUTE",
        "MED_RATE_LIMIT_INGEST_PER_MINUTE",
        "MED_RATE_LIMIT_REBUILD_PER_HOUR",
        "MED_SESSION_MAX_ENTRIES",
        "MED_SESSION_TTL_SECONDS",
        "MED_TRUSTED_HOSTS",
        "MOCK_LLM",
        "MODEL_WEIGHTS_OFFLINE",
        "PROJECT_ENV",
        "QIANFAN_AK",
        "QIANFAN_SK",
        "ALLOW_IMAGE_EGRESS",
        "ALLOW_TEXT_EGRESS",
        "OCR_API_BASE",
        "OCR_API_KEY",
        "OCR_API_MODEL",
        "OCR_API_TIMEOUT",
        "OCR_PROVIDER",
        "RERANK_API_BASE",
        "RERANK_API_KEY",
        "RERANK_API_TIMEOUT",
        "RERANK_PROVIDER",
        "RERANK_CACHE_DIR",
        "RERANK_ENABLED",
        "RERANK_MAX_LENGTH",
        "RERANKER_MODEL",
        "RERANK_TOP_N",
        "RETRIEVE_CANDIDATE_POOL",
        "RRF_K",
        "SAFETY_POLICY_PATH",
        "SAFETY_RESERVE_SLOTS",
        "SAFETY_SECTION_GATE_ENABLED",
        "SUFFICIENCY_DRY_RUN",
        "SUFFICIENCY_ENABLED",
        "SUFFICIENCY_MIN_DOC_COVERAGE",
        "SUFFICIENCY_MIN_UNION_COVERAGE",
        "ZHIPUAI_API_KEY",
    }
)


class EntrypointError(RuntimeError):
    """Fail-closed launcher error; messages never carry secret values."""


def _parse_secret_file(path: Path) -> dict[str, str]:
    """Parse a dotenv file as pure data. No shell is ever involved.

    ``interpolate=False`` keeps ``$(...)``/``${VAR}``/``$VAR`` sequences as
    literal characters, so a secret value can neither expand another value nor
    smuggle a command substitution into the environment.
    """
    try:
        if path.is_symlink() and not path.exists():
            raise EntrypointError(f"secret file is a broken symlink: {path}")
        if not path.is_file():
            raise EntrypointError(f"secret file is not a regular file: {path}")
        size = path.stat().st_size
    except OSError as exc:
        raise EntrypointError(f"secret file unreadable: {path} ({type(exc).__name__})") from exc
    if size > MAX_SECRET_BYTES:
        raise EntrypointError(f"secret file too large: {size} bytes (max {MAX_SECRET_BYTES})")
    # python-dotenv logs parse warnings that echo the offending raw line; a
    # secret value must never reach stderr, so silence its logger here.
    dotenv_logger = logging.getLogger("dotenv")
    previous_level = dotenv_logger.level
    dotenv_logger.setLevel(logging.CRITICAL)
    try:
        parsed: dict[str, Any] = dotenv_values(path, interpolate=False)
    except Exception as exc:  # parser failure: fail closed, report class only
        raise EntrypointError(f"secret file parse failed: {type(exc).__name__}") from exc
    finally:
        dotenv_logger.setLevel(previous_level)
    return {str(key).strip(): ("" if value is None else str(value)) for key, value in parsed.items() if str(key).strip()}


def _apply_allowlist(parsed: dict[str, str], environ: dict[str, str]) -> tuple[list[str], int]:
    """Export allowlisted keys only; return (applied names, ignored count)."""
    applied: list[str] = []
    ignored = 0
    for key, value in sorted(parsed.items()):
        if key in ALLOWLIST_ENV_KEYS:
            environ[key] = value
            applied.append(key)
        else:
            ignored += 1
    return applied, ignored


def _scrub_interpreter_hooks(environ: dict[str, str]) -> None:
    for key in INTERPRETER_HOOKS:
        environ.pop(key, None)


def _require_production_env(environ: dict[str, str]) -> None:
    """PROJECT_ENV must classify as production (W1.3 default-deny)."""
    # Lazy import on purpose: this is the only project import, and it happens
    # AFTER the secret file has been parsed and applied, so validation sees the
    # merged environment and the app-side config import happens only post-exec.
    from medcopilot.api.env_policy import classify_env, recognized_literals

    env_class = classify_env(environ.get("PROJECT_ENV"))
    if env_class != "production":
        raise EntrypointError(
            f"PROJECT_ENV must classify as production, got {env_class!r}; "
            f"accepted values: {', '.join(recognized_literals())}"
        )


def main(argv: list[str] | None = None) -> None:
    """Parse secrets, apply the allowlist, validate, then exec uvicorn."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) > 1:
        raise EntrypointError(f"usage: entrypoint [secret-file]; got {len(args)} arguments")
    secret_path = Path(args[0]) if args else Path(DEFAULT_SECRET_FILE)
    # ``exists()`` follows symlinks and answers False for a BROKEN one, so
    # gating on it alone would send a dangling secret mount down the ambient
    # fallback silently — the exact misconfiguration an operator must see.
    # Any symlink (broken or not) therefore goes through _parse_secret_file,
    # which fails closed on it; only a genuinely absent path falls back.
    if secret_path.is_symlink() or secret_path.exists():
        parsed = _parse_secret_file(secret_path)
        applied, ignored = _apply_allowlist(parsed, os.environ)
        # Names and counts only — never values.
        print(
            f"entrypoint: secret file {secret_path} parsed; "
            f"applied {len(applied)} allowlisted key(s), ignored {ignored} non-AppConfig key(s); "
            f"applied keys: {', '.join(applied)}",
            file=sys.stderr,
        )
    else:
        # Fallback path (documented): plain docker --env-file / -e / image ENV.
        print(
            f"entrypoint: secret file {secret_path} absent; using ambient environment",
            file=sys.stderr,
        )
    _scrub_interpreter_hooks(os.environ)
    _require_production_env(os.environ)
    print(
        f"entrypoint: exec {APP_MODULE} on {BIND_HOST}:{BIND_PORT} with workers={WORKERS}",
        file=sys.stderr,
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            APP_MODULE,
            "--host",
            BIND_HOST,
            "--port",
            str(BIND_PORT),
            "--workers",
            str(WORKERS),
        ],
    )


if __name__ == "__main__":
    try:
        main()
    except EntrypointError as exc:
        print(f"entrypoint: refusing to start: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except OSError as exc:  # exec itself failed (missing interpreter, etc.)
        print(f"entrypoint: exec failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(2) from None
    # os.execv never returns; reaching here means the process was not replaced.
    raise SystemExit(2) from None
