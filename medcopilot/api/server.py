from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from medcopilot.agents.ocr_api import build_ocr_tool
from medcopilot.agents.ocr_tool import OCRTool  # noqa: F401
from medcopilot.api.env_policy import classify_env, recognized_literals
from medcopilot.api.metrics import MetricsRegistry
from medcopilot.api.rate_limit import (
    ConcurrencyGuard,
    ConcurrencyLimitExceeded,
    RateLimitExceeded,
    SlidingWindowLimiter,
)
from medcopilot.api.request_context import (
    REQUEST_ID_HEADER,
    RequestContextMiddleware,
    get_request_id,
    sanitize_log_value,
)
from medcopilot.api.runtime import AppRuntime, ServingHolder, commit_serving, get_runtime
from medcopilot.api.session_store import SessionStore, SessionStoreFull
from medcopilot.audit import AuditContext, AuditLogger, derive_key_id, hash_session
from medcopilot.common.config import config
from medcopilot.common.logger import logger
from medcopilot.rag.clinical_chunker import ClinicalChunker, Document
from medcopilot.rag.corpus_store import CorpusStore
from medcopilot.rag.hybrid_retriever import HybridRetriever
from medcopilot.rag.index_manager import IndexManager
from medcopilot.rag.ingest_validator import IngestValidator
from medcopilot.rag.vector_store import ChromaDenseEncoder
from medcopilot.safety.policy import describe_health
from medcopilot.workflow.agent_graph import AgentGraph, ConsultationResult
from medcopilot.workflow.safety_guard import MedicalSafetyGuard

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB cap for drug-box photos
MAX_BODY_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024  # multipart form overhead margin
MAX_QUERY_CHARS = 2000
# Carried by every 429/503 the abuse controls emit: a throttled caller may be a
# real patient, so the rejection itself must never read as "wait for us".
OVERLOAD_EMERGENCY_NOTICE = "如出现胸痛、呼吸困难、意识不清等紧急情况，请立即拨打 120 或前往急诊，不要等待本服务。"

class BodySizeLimitMiddleware:
    """ASGI-level request-body cap.

    Runs BEFORE FastAPI's multipart parser, so oversized or chunked (no
    Content-Length) uploads are rejected at the transport layer instead of
    being fully spooled to disk first. Returns 413 as soon as the running
    total exceeds ``max_body_bytes``.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length and content_length.isdigit() and int(content_length) > self.max_body_bytes:
            await self._reject(send, "request body too large")
            return

        received = 0
        rejected = False
        response_started = False

        async def guarded_send(message: dict) -> None:
            """Suppress the framework's later 400 after we emitted a streamed 413."""
            nonlocal response_started
            if rejected:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def receive_limited() -> dict:
            nonlocal received, rejected
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    # Multipart parsing happens inside FastAPI after it calls receive().
                    # Return disconnect to halt that parser, emit exactly one 413 here,
                    # and suppress its later parse-error response via guarded_send.
                    if not response_started and not rejected:
                        rejected = True
                        await self._reject(send, "request body too large (streamed)")
                    return {"type": "http.disconnect"}
            return message

        await self.app(scope, receive_limited, guarded_send)

    async def _reject(self, send: Send, detail: str) -> None:
        """Emit exactly one 413 with an exact content-length and no keep-alive.

        剩余请求体不读、连接不复用（RFC 9112）：一旦拒收就断开连接，客户端
        不能继续向一个已被拒绝的请求流式传输，代理/客户端也不会在半途断掉的
        连接上复用残留字节。
        """
        body = f'{{"detail": "{detail}"}}'.encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"connection", b"close"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class ConsultationRequest(BaseModel):
    # Length is NOT capped in the schema: pydantic validation would reject
    # over-long bodies before the handler runs, and a long history pasted into
    # a 心梗 question must still reach the emergency short-circuit (the 120
    # answer is constant text with no data-leak surface). The handler enforces
    # the cap AFTER triage — see consult(). (The ASGI transport cap at
    # MAX_BODY_BYTES = 11 MiB remains the outer transport bound.)
    query: str = Field(...)
    # Optional opaque continuation token. Blank / missing => start a new
    # session; unknown / expired ids are a 404 for ordinary requests. Length
    # is enforced by SessionStore.validate_session_id inside the handler (not
    # in the schema) so schema validation cannot preempt the emergency
    # short-circuit — see consult(). Empty string is normalized to a new
    # session by the handler too.
    session_id: str | None = Field(default=None)


class ConsultationResponse(BaseModel):
    answer: str
    # Retrieved-evidence metadata entries. rerank_score 只在同一 rerank_mode
    # 内可比（cross-encoder 分数与词法重叠分数不同尺度），不是校准置信度；
    # 每条 entry 携带 rerank_mode 供消费方分组比较。
    sources: list[dict[str, Any]] = Field(
        ..., description="证据条目；rerank_score 只在同一 rerank_mode 内可比，不是校准置信度"
    )
    is_emergency: bool
    intent: str
    abstained: bool = False
    # Structured degradation signals ("rerank:lexical-fallback",
    # "llm:ConnectError", "ocr:inference_error", ...). Empty means every
    # serving component ran at full quality — the counterpart of /health for
    # the individual response, so an LLM outage is never mistaken for a
    # genuine evidence-insufficient abstention.
    degradations: list[str] = Field(default_factory=list)
    # OCR outcome for with-image consultations only: None on text-only
    # requests; otherwise "ok" | "image_unreadable" | "engine_unavailable" |
    # "inference_error". A photo the system failed to read must never present
    # as an ordinary text consultation.
    ocr_status: str | None = None
    # Every 200 response must carry a non-empty opaque id that legacy clients
    # can ignore; continuations send it back to reuse the last three turns.
    # An emergency short-circuit that had no live session returns an UNTRACKED
    # id (flood contract: the 120 path never creates session entries): a
    # follow-up on it lands on the ordinary 404 session-expired path, which
    # clients map to "start a new session" — nothing safety-relevant is lost,
    # the 120 answer is self-contained constant text.
    session_id: str
    # Version of the runtime safety-gating constants (code constants are the
    # source of truth) that produced this answer, so a historical response can
    # always be traced back to the policy version under which it was gated.
    policy_version: int = MedicalSafetyGuard.POLICY_VERSION
    # W6.1 dosage-gate stats: {"instructional", "redacted", "kept_quote",
    # "kept_risk"} — redaction frequency must be observable per response.
    dosage_gate: dict[str, int] = Field(default_factory=dict)


def retriever_degraded(runtime: AppRuntime) -> bool:
    retriever = runtime.retriever
    return bool(retriever is not None and getattr(retriever, "is_degraded", False))


def _has_valid_image_magic(data: bytes) -> bool:
    if data.startswith(b"\xff\xd8\xff"):  # JPEG
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n"):  # PNG
        return True
    if data.startswith(b"BM"):  # BMP
        return True
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":  # WebP
        return True
    return False


def create_app(
    initial_documents: list | None = None,
    clock=None,
    dense_encoder_factory=None,
) -> FastAPI:
    """Build the FastAPI app with an in-memory corpus and shared workflow.

    ``dense_encoder_factory`` runs inside ``ingest_lock`` (the boot binding and
    both rebuild embeddings), so an injected factory backed by a remote API
    serializes the rebuild path on that network call — keep injected encoders
    local or document the tradeoff.
    """
    from pathlib import Path

    monotonic_clock = clock or time.monotonic
    # Production env detection must precede app construction: the framework
    # docs/spec routes are not wrapped in require_reader (they are FastAPI
    # built-ins), so with the default configuration a production deployment
    # would expose its full API inventory via /docs and /openapi.json — the
    # textbook OWASP API8:2023 security-misconfiguration finding. FastAPI's
    # own conditional-openapi guidance: disable the spec, docs and redoc follow.
    # W1.3 (default-deny): only an explicit dev whitelist value opens the docs;
    # an unrecognized/empty PROJECT_ENV fails closed on reads/writes AND docs.
    env_class = classify_env(config.project_env)
    is_production = env_class == "production"
    if config.ocr_provider == "cloud":
        if not config.allow_image_egress:
            raise RuntimeError("OCR_PROVIDER=cloud requires ALLOW_IMAGE_EGRESS=true")
        if is_production and not config.ocr_api_base.lower().startswith("https://"):
            raise RuntimeError("cloud OCR requires an HTTPS OCR_API_BASE in production")
        if not config.ocr_api_base or not config.ocr_api_key or not config.ocr_api_model:
            raise RuntimeError("cloud OCR requires OCR_API_BASE, OCR_API_KEY and OCR_API_MODEL")
    for provider, enabled, base, key in (
        ("embedding", config.embedding_provider == "cloud", config.embedding_api_base, config.embedding_api_key),
        ("reranker", config.rerank_provider == "cloud", config.rerank_api_base, config.rerank_api_key),
    ):
        if enabled:
            if not config.allow_text_egress:
                raise RuntimeError(f"{provider} cloud mode requires ALLOW_TEXT_EGRESS=true")
            if is_production and not base.lower().startswith("https://"):
                raise RuntimeError(f"cloud {provider} requires an HTTPS endpoint in production")
            if not base or not key:
                raise RuntimeError(f"cloud {provider} requires an endpoint and API key")
    if config.llm_mode == "cloud":
        if not config.allow_text_egress:
            raise RuntimeError("cloud LLM mode requires ALLOW_TEXT_EGRESS=true")
        if is_production and not config.llm_api_base.lower().startswith("https://"):
            raise RuntimeError("cloud LLM requires an HTTPS LLM_API_BASE in production")
    docs_kwargs = {} if env_class == "dev" else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    # W5.6: version comes from AppConfig (APP_VERSION build-arg / env); the
    # git sha rides alongside it so a running instance can be matched against
    # the release evidence artifact. Never hard-coded here.
    app = FastAPI(
        title="Med-Copilot API",
        description="妙手回春临床辅助与智能用药问答服务（分诊 + 混合检索 RAG + 安全护栏）",
        version=config.app_version,
        **docs_kwargs,
    )
    # W5.4 ordering: RequestContextMiddleware is added LAST so it is the
    # OUTERMOST user middleware — every response the inner layers emit
    # (413 body-limit, 401/503 auth, 429/503 abuse controls, TrustedHost
    # rejections) passes back through it and carries X-Request-ID. A rejected
    # caller still needs a correlation id to quote when they call the clinic.
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=MAX_BODY_BYTES)
    if config.cors_allow_all and env_class == "dev":
        # Development escape-hatch: allow wildcard origins, but per CORS spec
        # allow_credentials MUST be False with wildcard to avoid browser rejections
        # and credential leakage across arbitrary domains. W1.3: the wildcard is
        # honored only for an explicit dev environment — production/unknown
        # deployments always fall back to the strict origin list.
        cors_kwargs = {"allow_origins": ["*"], "allow_credentials": False}
    else:
        cors_kwargs = {
            "allow_origins": [o.strip() for o in config.cors_origins.split(",") if o.strip()],
            "allow_credentials": True,
        }
    app.add_middleware(
        CORSMiddleware,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
        # W5.4: expose the correlation header to browsers (default CORS policy
        # hides non-safelisted response headers from JS, so the Gradio client
        # could not surface an id in an error report without this).
        expose_headers=[
            REQUEST_ID_HEADER,
            "Retry-After",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
        **cors_kwargs,
    )
    # Host-header allowlist (host-header injection / password-reset poisoning
    # mitigation). Comma-separated config; "*" (the default) keeps existing
    # zero-config deployments working. Added last so it sits OUTSIDE CORS and
    # the body limiter — a forged Host is rejected before anything else runs.
    allowed_hosts = [h.strip() for h in config.trusted_hosts.split(",") if h.strip()] or ["*"]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    # W5.6 zero-dependency metrics registry (see api/metrics.py). Process-local
    # like every other runtime state: the deployment is single-worker.
    metrics = MetricsRegistry()
    app.state.metrics = metrics
    boot_monotonic = time.monotonic()
    app.state.boot_monotonic = boot_monotonic
    app.add_middleware(RequestContextMiddleware, metrics=metrics)

    documents: list[Document] = list(initial_documents or [])
    # Authoritative corpus (Slice 2): when persistence is configured and no
    # documents were injected programmatically, the process boots from the
    # on-disk authority. Injected documents (tests/embedding) bypass the disk
    # entirely, preserving the pre-Slice-2 behavior for that path.
    injected_documents = bool(initial_documents)
    corpus_store: CorpusStore | None = None
    if config.med_corpus_path:
        corpus_store = CorpusStore(path=config.med_corpus_path)
        if not injected_documents:
            documents = corpus_store.load()
            if corpus_store.degraded:
                logger.warning(
                    "Authoritative corpus at {} has skipped records; serving degraded.",
                    config.med_corpus_path,
                )

    # Build the dense encoder FIRST so the single retriever that state/workflow
    # hold is the dense-wired one. A prior version created a dense-less retriever
    # here and then built a second dense retriever into a throwaway local that was
    # never bound to state/workflow, so dense retrieval was dead on the request
    # path until an ingestion happened to rebuild it.
    # Slice 3: the factory is injectable for tests; with a manifest present the
    # process rebinds to the blue/green collection the manifest designates.
    encoder_factory = dense_encoder_factory or (
        lambda name: ChromaDenseEncoder.build_encoder(
            persist_directory=config.chroma_persist_directory, collection_name=name
        )
    )
    dense_encoder = None
    index_manager: IndexManager | None = None
    if config.enable_chroma_dense:
        if corpus_store is not None:
            index_manager = IndexManager(
                corpus_store=corpus_store,
                encoder_factory=encoder_factory,
                manifest_path=Path(config.med_corpus_path).parent / "index_manifest.json",
                base_collection="med_copilot",
            )
        active_name = (
            index_manager.current["collection"] if index_manager and index_manager.current else "med_copilot"
        )
        dense_encoder = encoder_factory(active_name)
        if index_manager is not None and dense_encoder is not None:
            index_manager.bind_active(dense_encoder, active_name)
            # Boot-time divergence warning: a corpus that has moved past the
            # manifest (crash between the durable corpus commit and the
            # best-effort Chroma mirror, or out-of-band edits) is logged here
            # and reported LIVE by ``_index_stale_now`` below — a boot constant
            # used to be reported instead and never recovered after a rebuild.
            manifest = index_manager.current
            if manifest and index_manager.corpus_hash(documents) != manifest.get("corpus_sha256"):
                logger.warning(
                    "Authoritative corpus hash diverges from the manifest pin ({}); "
                    "dense index is stale — run POST /index/rebuild.",
                    (manifest.get("corpus_sha256") or "")[:12],
                )
        if dense_encoder is not None:
            logger.info("Chroma dense encoder enabled (collection='{}').", active_name)

    retriever = HybridRetriever(documents=documents, k=config.rrf_k, dense_encoder=dense_encoder)
    workflow = AgentGraph(retriever=retriever)
    ocr = build_ocr_tool()
    # W1.2 medical audit sink: append-only JSONL + SHA-256 hash chain, PHI-free.
    # An audit failure must never affect the request path (failures are counted
    # and surfaced on /health instead of raised).
    audit = AuditLogger(
        config.audit_log_path,
        retention_days=config.audit_retention_days,
        fsync=config.audit_fsync,
        ntp_server=config.audit_ntp_server,
    )
    audit_key_id = derive_key_id((config.admin_api_key or "").strip())
    # Frozen serving snapshot (api/runtime.py): replaces the previous mutable
    # ``state`` dict whose keys were written one-by-one during ingest/rebuild.
    # Boot generation is constructed directly — every component is final here,
    # and the workflow was already wired to this retriever in its constructor.
    runtime_holder = ServingHolder(
        AppRuntime(
            documents=documents,
            retriever=retriever,
            dense_encoder=dense_encoder,
            ocr=ocr,
            workflow=workflow,
            audit=audit,
        )
    )
    app.state.runtime_holder = runtime_holder
    ingest_lock = threading.Lock()

    # Live index staleness (wave-10): the boot-time ``index_stale`` above is a
    # snapshot of the boot corpus and cannot see later ingests/rebuilds, so
    # /health used to report a frozen value while /index/status computed the
    # same fact live — after a successful rebuild the two surfaces disagreed
    # and readiness never recovered. The check is now the CURRENT manifest vs
    # the CURRENT serving corpus (the expression /index/status reports as
    # ``corpus_matches_index``, negated), memoized per (serving generation,
    # manifest): commit_serving publishes a new AppRuntime with a single
    # reference write and IndexManager replaces its manifest dict, so identity
    # keys catch every change without re-hashing the corpus on every poll.
    # Dense disabled / no manifest ⇒ not stale (absent, not failed).
    _stale_cache: dict[str, Any] = {"entry": None}

    def _index_stale_now() -> bool:
        serving = runtime_holder.runtime
        manifest = index_manager.current if index_manager is not None else None
        if manifest is None:
            return False
        key = (serving, manifest)
        entry = _stale_cache["entry"]
        if entry is not None and entry[0][0] is serving and entry[0][1] is manifest:
            return entry[1]
        stale = index_manager.corpus_hash(serving.documents) != manifest.get("corpus_sha256")
        _stale_cache["entry"] = (key, stale)  # single atomic reference write
        return stale

    # Process-local bounded session store (single-worker deployment). A lock per
    # session serializes that session's read->workflow->record sequence without
    # holding any global lock across OCR/retrieval/LLM calls; the store owns the
    # locks' lifetime so evicted sessions cannot leak them.
    sessions = SessionStore(
        ttl_seconds=config.med_session_ttl_seconds,
        max_entries=config.med_session_max_entries,
        clock=monotonic_clock,
    )
    app.state.sessions = sessions

    # ------------------------------------------------------------------ auth
    admin_token = (config.admin_api_key or "").strip()
    # ``env_class`` was computed before app construction (docs gating);
    # quoted .env values keep their whitespace: PROJECT_ENV=" production "
    # must still be recognized as production or the fail-closed net silently
    # opens.
    # W1.3 default-deny: reads/writes open ONLY for an explicit dev whitelist
    # value with no key configured. Production → 503 without a key (never
    # fail-open). Unrecognized (empty / typo) → 503 even WITH a key: an env
    # that cannot be classified must never be guessed into an open state.
    reads_open = not admin_token and env_class == "dev"
    require_auth = bool(admin_token) or env_class != "dev"

    def _extract_bearer(request: Request) -> str | None:
        authorization = request.headers.get("Authorization", "")
        scheme, param = get_authorization_scheme_param(authorization)
        return param if scheme.lower() == "bearer" else None

    def require_reader(request: Request) -> None:
        if reads_open:
            return
        if env_class == "unrecognized":
            # Fail closed for an unclassifiable environment, key or no key.
            _record_auth_failure(request.url.path, 503)
            raise HTTPException(
                status_code=503,
                detail=(
                    "PROJECT_ENV not recognized; set it to one of "
                    f"{recognized_literals()} (reads stay closed until then)"
                ),
            )
        if not admin_token:
            # Production with no key configured: never fall open on reads.
            _record_auth_failure(request.url.path, 503)
            raise HTTPException(
                status_code=503, detail="service unavailable: ADMIN_API_KEY not configured"
            )
        provided = _extract_bearer(request)
        # Compare bytes: compare_digest rejects non-ASCII str with TypeError,
        # which would turn a malformed bearer token into a 500 instead of a 401.
        if not provided or not secrets.compare_digest(
            provided.encode("utf-8"), admin_token.encode("utf-8")
        ):
            _record_auth_failure(request.url.path, 401)
            raise HTTPException(status_code=401, detail="missing or invalid API key")

    def require_admin(request: Request) -> None:
        if env_class == "unrecognized":
            _record_auth_failure(request.url.path, 503)
            raise HTTPException(
                status_code=503,
                detail=(
                    "PROJECT_ENV not recognized; set it to one of "
                    f"{recognized_literals()} (ingestion stays closed until then)"
                ),
            )
        if not admin_token:
            # No admin key configured: knowledge-base writes stay open only for
            # explicit local dev; production deployments must set ADMIN_API_KEY.
            if is_production:
                _record_auth_failure(request.url.path, 503)
                raise HTTPException(status_code=503, detail="ingestion disabled: ADMIN_API_KEY not configured")
            return
        provided = _extract_bearer(request)
        # Bytes comparison — see require_reader.
        if not provided or not secrets.compare_digest(
            provided.encode("utf-8"), admin_token.encode("utf-8")
        ):
            _record_auth_failure(request.url.path, 401)
            raise HTTPException(status_code=401, detail="missing or invalid admin API key")

    # ------------------------------------------------- abuse controls (W2.3)
    # Both guards sit AFTER emergency triage, auth and request validation in
    # every handler: a 120 red-line warning is never throttled, and an
    # oversized body is already a 413 (BodySizeLimitMiddleware) before any
    # quota is spent. Process-local by design (single worker, see README).
    rate_limiters: dict[str, SlidingWindowLimiter] = {}
    if config.rate_limit_enabled:
        for path, limit, window in (
            ("/consultations", config.rate_limit_consult_per_minute, 60.0),
            ("/consultations/with-image", config.rate_limit_image_per_minute, 60.0),
            ("/documents/drug-label", config.rate_limit_ingest_per_minute, 60.0),
            ("/index/rebuild", config.rate_limit_rebuild_per_hour, 3600.0),
        ):
            rate_limiters[path] = SlidingWindowLimiter(limit, window, clock=monotonic_clock)
    expensive_paths = ConcurrencyGuard(config.llm_concurrency)

    # ------------------------------------------------ emergency-flood budget
    # The pre-auth emergency short-circuit is unauthenticated AND unthrottled
    # by contract (a 120 red-line warning is never delayed or rejected), so
    # its per-request side effects must not scale with a flood: 200
    # unauthenticated emergency requests used to write 200 audit rows and
    # create 200 session entries, evicting real patient sessions — and the
    # /health abuse-control design explicitly promises "a flood cannot turn
    # the audit sink into a disk-fill vector". Two independent meters fix that
    # without touching the answer itself:
    # * session writes: a short-circuit never CREATES a session (see
    #   _emergency_response) — an existing session still gets its turn;
    # * audit rows: individual rows only up to the ordinary /consultations
    #   quota per window — an existing, operator-tuned budget, not a new
    #   magic number — beyond it the flood is counted here and audited once
    #   per window as an aggregate row (still fixed 120 text, always).
    emergency_budget = max(1, config.rate_limit_consult_per_minute)
    emergency_window = (
        rate_limiters["/consultations"].window_seconds
        if "/consultations" in rate_limiters
        else 60.0  # same window the consultation limiter uses when it is off
    )
    emergency_flood_lock = threading.Lock()
    emergency_flood = {"window_start": monotonic_clock(), "recorded": 0, "suppressed": 0}

    def _charge_emergency_flood() -> tuple[bool, int]:
        """Count one emergency triage; return (individual_row_due, pending_aggregate).

        ``pending_aggregate`` is non-zero exactly when a window with
        suppressed triages has just rolled over and still owes its single
        summary row — the caller writes it before its own row so the trail
        stays chronological.
        """
        with emergency_flood_lock:
            now = monotonic_clock()
            pending = 0
            if now - emergency_flood["window_start"] >= emergency_window:
                pending = emergency_flood["suppressed"]
                emergency_flood["window_start"] = now
                emergency_flood["recorded"] = 0
                emergency_flood["suppressed"] = 0
            if emergency_flood["recorded"] < emergency_budget:
                emergency_flood["recorded"] += 1
                return True, pending
            emergency_flood["suppressed"] += 1
            return False, pending

    def _emergency_flood_suppressed() -> int:
        """Live suppressed count (observability only: /health + /metrics)."""
        with emergency_flood_lock:
            return emergency_flood["suppressed"]

    auth_flood_lock = threading.Lock()
    auth_flood = {"window_start": monotonic_clock(), "recorded": 0, "suppressed": 0}

    def _record_auth_failure(endpoint: str, status_code: int) -> None:
        with auth_flood_lock:
            now = monotonic_clock()
            pending = 0
            if now - auth_flood["window_start"] >= emergency_window:
                pending = auth_flood["suppressed"]
                auth_flood["window_start"] = now
                auth_flood["recorded"] = 0
                auth_flood["suppressed"] = 0
            if auth_flood["recorded"] < emergency_budget:
                auth_flood["recorded"] += 1
                individual = True
            else:
                auth_flood["suppressed"] += 1
                individual = False
        ctx = AuditContext(endpoint=endpoint, key_id=audit_key_id)
        if pending:
            _audit_event(ctx, "auth_failure_flood", status_code=status_code, aggregated=pending)
        if individual:
            _audit_event(ctx, "auth_failure", status_code=status_code)

    def _throttle(request: Request, endpoint: str) -> None:
        """Charge one request against the endpoint quota of the caller's identity.

        Identity is the API key (the request already passed auth) or, only in
        no-key dev mode, the client address. 429 carries Retry-After and the
        X-RateLimit-* headers plus an ``emergency_notice`` in the body.
        ``X-RateLimit-Reset`` is the ABSOLUTE epoch second at which the window
        frees (IETF ratelimit-headers semantics); ``Retry-After`` and
        ``retry_after_seconds`` stay relative durations — mixing the two
        conventions is how a client ends up computing a 1970 reset.
        """
        limiter = rate_limiters.get(endpoint)
        if limiter is None:
            return
        identity = f"key:{audit_key_id}" if admin_token else f"ip:{request.client.host if request.client else '?'}"
        try:
            limiter.check(identity)
        except RateLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limited",
                    "retry_after_seconds": exc.retry_after,
                    "limit": exc.limit,
                    "window_seconds": int(limiter.window_seconds),
                    "emergency_notice": OVERLOAD_EMERGENCY_NOTICE,
                },
                headers={
                    "Retry-After": str(exc.retry_after),
                    "X-RateLimit-Limit": str(exc.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + exc.retry_after),
                },
            ) from exc

    @contextmanager
    def _heavy_slot() -> Iterator[None]:
        """Hold one in-flight slot for ANY expensive work, or fail fast with 503.

        Covers the consult/OCR paths (retrieval + rerank + LLM) AND the admin
        ingest/rebuild paths while they park on ``ingest_lock``: a waiter holds
        its slot for the whole wait, so stalled rebuilds consume the bounded
        budget instead of parking unbounded threads in the shared pool (the
        classic thread-starvation path that also starves emergency triage).
        Rejection never waits: parked threads are what starve emergency
        triage and /health when the LLM upstream hangs.
        """
        try:
            with expensive_paths.slot():
                yield
        except ConcurrencyLimitExceeded as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "server_busy",
                    "retry_after_seconds": 1,
                    "limit": exc.limit,
                    "emergency_notice": OVERLOAD_EMERGENCY_NOTICE,
                },
                headers={"Retry-After": "1"},
            ) from exc

    @contextmanager
    def _ingest_region() -> Iterator[None]:
        """``ingest_lock`` critical region with its waiters guard-counted.

        Every holder already owns a ``_heavy_slot`` (bounding the parked
        threads); this adds the waiter observability so lock congestion shows
        up in /health and /metrics before it degrades into starvation.
        """
        with expensive_paths.waiter():
            with ingest_lock:
                yield

    def _llm_health_fields() -> dict[str, Any]:
        llm = runtime_holder.runtime.workflow.llm
        classifier = getattr(llm, "classify_mode", None)
        if classifier is None:
            return {
                "llm_mode": "error",
                "llm_mode_reason": "status_unavailable",
                "llm_model_effective": "unknown",
                "llm_mock_fallback_total": 0,
                "llm_warnings": ["status_unavailable"],
                "llm_provider": config.llm_provider,
                "embedding_provider": config.embedding_provider,
                "rerank_provider": config.rerank_provider,
                "ocr_provider": config.ocr_provider,
                "image_egress_allowed": bool(config.allow_image_egress),
            }
        mode, reason = classifier(
            getattr(llm, "provider", config.llm_provider),
            getattr(llm, "api_key", config.llm_api_key),
            getattr(llm, "api_base", config.llm_api_base),
            bool(getattr(llm, "_mock_enabled", lambda: config.mock_llm)()),
        )
        model = getattr(llm, "model_name", config.llm_model)
        if mode == "mock":
            effective_model = f"mock:{model}"
        elif mode == "error":
            effective_model = "mock-blocked"
        else:
            effective_model = model
        # W2.5: provider↔key 形态问题的稳定码(静态检查,只 warn 不改行为)。
        warnings_getter = getattr(llm, "config_warnings", None)
        llm_warnings = sorted(warnings_getter()) if callable(warnings_getter) else []
        return {
            "llm_mode": mode,
            "llm_mode_reason": reason,
            "llm_model_effective": effective_model,
            "llm_mock_fallback_total": int(getattr(llm, "mock_fallback_total", 0)),
            "llm_warnings": llm_warnings,
            "llm_provider": getattr(llm, "provider", config.llm_provider),
            "embedding_provider": config.embedding_provider,
            "rerank_provider": config.rerank_provider,
            "ocr_provider": config.ocr_provider,
            "image_egress_allowed": bool(config.allow_image_egress),
        }

    def _health_components(full: dict[str, Any]) -> dict[str, Any]:
        """Component readiness view (W5.6) computed from the existing payload.

        No new probes and no disk/model work on the request path — every value
        is a field the payload already derived from the public runtime APIs
        (IndexManager manifest/pin attrs, retriever ``is_degraded``, reranker
        ``mode``, audit sink status). Semantics:

        - ``index.pin_verified`` is ``None`` when dense is disabled (absent,
          not failed), ``False`` when the manifest pin could NOT be verified,
          ``True`` only when verified. It is never coerced to green.
        - ``reranker.weights`` reports the reranker's ``weights_state``
          (``disabled`` | ``unloaded`` | ``local`` | ``missing`` | ``cloud``);
          a missing attribute reads as ``None``, never as OK.
        """
        return {
            "corpus": {
                "loaded": int(full.get("corpus_documents", 0)) > 0,
                "degraded": bool(full.get("corpus_degraded")),
                "documents": int(full.get("corpus_documents", 0)),
                "persisted": bool(full.get("corpus_persisted")),
            },
            "index": {
                "enabled": index_manager is not None,
                "manifest": full.get("index_manifest"),
                "pin_verified": full.get("index_pin_verified"),
                "stale": bool(full.get("index_stale")),
                "dense_degraded": bool(full.get("dense_degraded")),
            },
            "reranker": {
                "mode": full.get("rerank_mode"),
                "weights": getattr(
                    runtime_holder.runtime.workflow.reranker, "weights_state", None
                ),
            },
            "llm": {
                "mode": full.get("llm_mode"),
                "reason": full.get("llm_mode_reason"),
                "model_effective": full.get("llm_model_effective"),
            },
            "policy": {
                "version": full.get("policy_version"),
                "matches_code": full.get("policy_matches_code"),
                "review_status": full.get("policy_review_status"),
                "clinical_signoff": bool(full.get("clinical_signoff", False)),
            },
            "audit": {
                "ready": bool(full.get("audit_ready")),
                "chain_verified": bool(full.get("audit_chain_verified")),
                "failures": int(full.get("audit_failures", 0)),
            },
            "abuse_controls": {
                "rate_limit_enabled": bool(full.get("rate_limit_enabled")),
                "rate_limited_total": int(full.get("rate_limited_total", 0)),
                "overloaded_total": int(full.get("overloaded_total", 0)),
                "emergency_flood_suppressed": int(full.get("emergency_flood_suppressed", 0)),
                "in_flight": int(full.get("llm_in_flight", 0)),
                "waiters": int(full.get("llm_waiters", 0)),
                "limit": full.get("llm_concurrency_limit"),
            },
        }

    def _compute_ready(components: dict[str, Any]) -> bool:
        """Readiness (NOT liveness): can this instance serve a grounded answer?

        - corpus loaded and not degraded (a corrupt/empty corpus is not ready);
        - dense index, when enabled, verified against its manifest pin — an
          UNVERIFIED pin (False) or a stale one is NOT ready: no fake green;
        - audit sink healthy (ready, no failures, chain verified at startup).

        HTTP status stays 200 with ``ready=false``: health semantics and HTTP
        semantics are decoupled — a reverse proxy may drain traffic on
        ``ready=false`` if the deployment wants 503-style behaviour.
        """
        corpus = components["corpus"]
        if not corpus["loaded"] or corpus["degraded"]:
            return False
        index = components["index"]
        if index["enabled"] and (index["stale"] or index["pin_verified"] is not True):
            return False
        audit = components["audit"]
        if not audit["ready"] or not audit["chain_verified"]:
            return False
        return True

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        """Liveness/readiness probe with an auth-tiered payload.

        Full payload for ``reads_open`` configurations or a valid Bearer (same
        comparison mechanics as ``require_reader`` but never 401/503 — an
        anonymous caller only gets a reduced payload). Otherwise just the
        liveness core: ``status/auth_required/env_*`` plus the policy-mirror
        governance fields from ``describe_health`` (policy version, drift,
        review and signoff status — governance facts, not operational
        inventory). Rationale: OWASP API8:2023 — health endpoints must not
        leak the deployment's operational inventory (corpus/index/rerank/OCR/
        audit details) to unauthenticated scanners, while Docker HEALTHCHECK
        (Dockerfile:95-96) still needs an anonymous 200, and the no-key dev
        mode used by recovery_drill / load_check / the test suite is unchanged
        (reads_open => full payload).
        生产无密钥时 ``auth_required=true`` 即误配的可见信号。
        """
        full = {
            "status": "ok",
            "corpus_documents": len(runtime_holder.runtime.documents),
            "corpus_persisted": corpus_store is not None,
            "corpus_degraded": bool(corpus_store is not None and corpus_store.degraded),
            # Safety-policy mirror status: a broken/missing artifact reports
            # honestly here but never weakens the actual code-constant gating.
            **describe_health(config.safety_policy_path),
            "ocr_available": runtime_holder.runtime.ocr.is_available,
            "dense_degraded": retriever_degraded(runtime_holder.runtime),
            # Process-level rerank mode: "cross-encoder" | "lexical" |
            # "disabled" — a CrossEncoder that failed to load must not leave
            # /health all-green while evidence ordering quietly degrades.
            "rerank_mode": runtime_holder.runtime.workflow.reranker.mode,
            # Boot-time divergence is only the startup warning; the reported
            # value is live so a post-boot ingest/rebuild cannot leave /health
            # (and readiness) frozen at the boot-time answer.
            "index_stale": _index_stale_now(),
            # Dense-index manifest semantics aligned with the policy artifact:
            # "ok" | "absent" | "invalid" | "disabled". A corrupted manifest is
            # NOT the same as no manifest — it must be visible here.
            "index_manifest": (
                index_manager.manifest_status if index_manager is not None else "disabled"
            ),
            # Boot-time pin health probe: True only when the bound collection
            # matched the manifest AND its live chunk count equals the pin.
            "index_pin_verified": index_manager.pin_verified if index_manager is not None else None,
            "auth_required": require_auth,
            # W1.3 default-deny surface: an unrecognized/empty PROJECT_ENV is
            # visible here (it never silently opens reads/writes).
            "env_class": env_class,
            "env_recognized": env_class != "unrecognized",
            # W1.2 audit surface: an audit sink that cannot write is visible
            # here (failures>0 ⇒ audit_ready=false); the chain head anchors the
            # log externally (release_pack) so deletions are detectable.
            "audit_ready": audit.ready and audit.failures == 0,
            "audit_sink": audit.path or None,
            "audit_records_today": audit.records_today,
            "audit_failures": audit.failures,
            "audit_chain_head": audit.chain_head(),
            "audit_chain_verified": audit.chain_verified,
            "audit_chain_broken_at": audit.chain_broken_at,
            "audit_startup_error": audit.startup_error,
            "audit_retention_days": audit.retention_days,
            "audit_time_offset_ms": audit.time_offset_ms(),
            # W2.3 abuse-control surface (process-lifetime totals): rejections
            # are counted here instead of audited one by one, so a flood cannot
            # turn the audit sink into a disk-fill vector.
            "rate_limit_enabled": config.rate_limit_enabled,
            "rate_limited_total": sum(limiter.rejected for limiter in rate_limiters.values()),
            "overloaded_total": expensive_paths.rejected,
            # Emergency short-circuits are unauthenticated and unthrottled by
            # contract; beyond the flood budget their audit rows are aggregated
            # (one summary row per window) and this counter keeps the suppressed
            # total visible — the same "count, never audit one-by-one" rule.
            "emergency_flood_suppressed": _emergency_flood_suppressed(),
            "llm_concurrency_limit": expensive_paths.limit,
            "llm_in_flight": expensive_paths.in_flight,
            "llm_waiters": expensive_paths.waiters,
            "sessions_active": sessions.size,
            "sessions_capacity": config.med_session_max_entries,
            # W5.6 release identity: matched against the release evidence
            # artifact so a running instance is always attributable to a commit.
            "version": config.app_version,
            "git_sha": config.app_git_sha,
            **_llm_health_fields(),
        }
        components = _health_components(full)
        full["components"] = components
        full["ready"] = _compute_ready(components)
        if reads_open:
            return full
        provided = _extract_bearer(request)
        # Bytes comparison — see require_reader (same rule, minus the raises).
        if (
            admin_token
            and provided
            and secrets.compare_digest(provided.encode("utf-8"), admin_token.encode("utf-8"))
        ):
            return full
        # Anonymous + closed reads: liveness core only (always HTTP 200 so the
        # orchestrator HEALTHCHECK keeps passing).
        return {
            "status": "ok",
            "auth_required": require_auth,
            "env_class": env_class,
            "env_recognized": env_class != "unrecognized",
            **describe_health(config.safety_policy_path),
        }

    @app.get("/health/live")
    def health_live() -> dict[str, Any]:
        """Liveness (W5.6): the process is up. Nothing else.

        ZERO dependencies and ZERO auth by design — the Docker HEALTHCHECK
        probes this from inside the container without a Bearer token, and a
        corpus/index/policy degradation must never restart the container: no
        disk reads, no model probes, no component inspection. Payload is
        deliberately minimal (no version, no inventory — that is the
        auth-tiered ``/health`` readiness surface's job).
        """
        return {
            "status": "alive",
            "uptime_seconds": round(time.monotonic() - boot_monotonic, 1),
        }

    @app.get("/metrics", dependencies=[Depends(require_admin)])
    def metrics_endpoint() -> Response:
        """Prometheus exposition (W5.6), admin-authenticated.

        Counters/histograms accumulate on the request path; gauges are set at
        scrape time from the SAME observables /health reports (corpus, rate
        limits, concurrency/waiters, sessions, audit, degradations), so the
        two surfaces cannot disagree. Label set is non-sensitive only:
        endpoint, status, component, mode, intent — no query text, no host,
        no client address, no session id, no key material.
        """
        _refresh_gauges()
        return Response(
            metrics.export_prom(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    def _refresh_gauges() -> None:
        """Set-mode gauges from process observables at scrape time."""
        try:
            serving = runtime_holder.runtime
            metrics.set("med_rag_corpus_documents", len(serving.documents))
            metrics.set(
                "med_rag_corpus_degraded",
                1 if (corpus_store is not None and corpus_store.degraded) else 0,
            )
            metrics.set("med_rag_sessions_active", sessions.size)
            metrics.set("med_rag_expensive_in_flight", expensive_paths.in_flight)
            metrics.set("med_rag_expensive_waiters", expensive_paths.waiters)
            metrics.set(
                "med_rag_rate_limited_total",
                sum(limiter.rejected for limiter in rate_limiters.values()),
            )
            metrics.set("med_rag_overloaded_total", expensive_paths.rejected)
            metrics.set("med_rag_emergency_flood_suppressed", _emergency_flood_suppressed())
            metrics.set("med_rag_index_stale", 1 if _index_stale_now() else 0)
            pin = index_manager.pin_verified if index_manager is not None else None
            # -1 = dense disabled (no opinion), 0 = unverified/invalid, 1 = verified
            metrics.set("med_rag_index_pin_verified", {True: 1, False: 0, None: -1}[pin])
            metrics.set("med_rag_llm_mode", _mode_to_number(_llm_health_fields().get("llm_mode", "")))
            metrics.set("med_rag_rerank_mode", _mode_to_number(serving.workflow.reranker.mode))
            metrics.set_counter("med_rag_audit_write_failures_total", audit.failures)
            metrics.set(
                "med_rag_build_info",
                1,
                labels={"version": config.app_version, "git_sha": config.app_git_sha},
            )
        except Exception as exc:  # pragma: no cover - observability must never 500
            logger.warning("gauge refresh failed ({}); /metrics may be stale", exc)

    @staticmethod
    def _mode_to_number(mode: str) -> float:
        # Numeric projection for alerting (mock/lexical ≠ full quality):
        # live/cross-encoder/cloud=2, mock/lexical=1, anything else (error/disabled/
        # unavailable)=0. Modes stay string-typed in /health for humans.
        return {"live": 2.0, "cross-encoder": 2.0, "cloud": 2.0, "mock": 1.0, "lexical": 1.0}.get(mode, 0.0)

    @app.post("/documents/drug-label", dependencies=[Depends(require_admin)])
    def ingest_drug_label(
        request: Request,
        raw_text: str = Form(...),
        drug_name: str = Form("未知药品"),
    ) -> dict[str, Any]:
        """Parse a drug label into clinical sections and index it for retrieval."""
        _throttle(request, "/documents/drug-label")
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="raw_text must not be empty")
        if len(raw_text) > config.max_drug_label_chars:
            raise HTTPException(
                status_code=413, detail=f"raw_text too long (max {config.max_drug_label_chars} chars)"
            )
        raw_drug_name = drug_name or ""
        if not " ".join(raw_drug_name.split())[:200]:
            raise HTTPException(status_code=422, detail="drug_name must not be blank")
        # W3.2 ingest gate (OWASP LLM08): validate BEFORE normalizing/truncating
        # drug_name and BEFORE chunking/indexing — control characters or tail
        # injection payloads in drug_name must be rejected with 422 and audited,
        # not silently scrubbed away by whitespace collapse/truncation.
        if config.ingest_validation_enabled:
            verdict = IngestValidator.validate(
                raw_text, raw_drug_name,
                min_raw_chars=config.ingest_min_raw_chars,
                max_line_chars=config.ingest_max_line_chars,
            )
            if not verdict.accepted:
                audit.record(
                    event_type="ingest_rejected",
                    endpoint="/documents/drug-label",
                    session_hash="anonymous",
                    key_id=audit_key_id,
                    intent="ingest",
                    is_emergency=False,
                    red_rule_ids=(),
                    abstained=False,
                    degradations=tuple(f"ingest:{reason}" for reason in verdict.reasons),
                    evidence_chunk_ids=(),
                    evidence_count=0,
                    policy_version=MedicalSafetyGuard.POLICY_VERSION,
                    latency_ms=0.0,
                    status_code=422,
                )
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": "drug label failed ingest validation",
                        "reasons": list(verdict.reasons),
                        "raw_text_sha256": verdict.raw_sha256,
                    },
                )
        # Collapse whitespace before the length cap: the name flows into chunk
        # metadata, content-addressed chunk_id hashing, page_content prefixes
        # and API echoes, and CorpusStore.upsert_drug matches old chunks by the
        # stripped name — inconsistent stripping would strand stale chunks.
        drug_name = " ".join(raw_drug_name.split())[:200]
        docs = ClinicalChunker.chunk_drug_label(raw_text, drug_name=drug_name)
        # Lock guards read-modify-write: concurrent ingestions must not double-insert
        # chunks or leave the serving snapshot pointing at a stale retriever.
        # The whole region runs inside _heavy_slot (via _ingest_region): lock
        # waiters are bounded by the concurrency guard, not by the thread pool.
        with _heavy_slot():
            with _ingest_region():
                serving = runtime_holder.runtime
                if corpus_store is not None and not injected_documents:
                    # Slice 2 idempotent path: the on-disk authority is replaced for
                    # this drug atomically (write-through), then the serving cache
                    # and retriever are rebuilt from the committed snapshot.
                    chunks_added, chunks_replaced = corpus_store.upsert_drug(drug_name, docs)
                    new_documents = corpus_store.snapshot
                    new_retriever = HybridRetriever(
                        documents=new_documents, k=config.rrf_k, dense_encoder=serving.dense_encoder
                    )
                    commit_serving(
                        runtime_holder,
                        documents=new_documents,
                        retriever=new_retriever,
                        dense_encoder=serving.dense_encoder,
                        ocr=serving.ocr,
                        workflow=serving.workflow,
                        audit=serving.audit,
                    )
                else:
                    # Legacy in-memory path (persistence disabled or injected docs):
                    # dedup by chunk_id and append.
                    existing_ids = {d.metadata.get("chunk_id") for d in serving.documents}
                    new_docs = [d for d in docs if d.metadata.get("chunk_id") not in existing_ids]
                    chunks_added, chunks_replaced = len(new_docs), 0
                    if new_docs:
                        new_documents = [*serving.documents, *new_docs]
                        new_retriever = HybridRetriever(
                            documents=new_documents, k=config.rrf_k, dense_encoder=serving.dense_encoder
                        )
                        commit_serving(
                            runtime_holder,
                            documents=new_documents,
                            retriever=new_retriever,
                            dense_encoder=serving.dense_encoder,
                            ocr=serving.ocr,
                            workflow=serving.workflow,
                            audit=serving.audit,
                        )
            # NOTE: ingestion deliberately does NOT touch the persistent Chroma
            # collections. ``index_documents`` is a full-replace primitive whose
            # only legitimate caller is the blue-green rebuild (IndexManager);
            # mirroring incremental chunks here would truncate the active
            # collection to the last drug. Collections are disposable
            # derivatives of the authoritative corpus — POST /index/rebuild is
            # their single writer, and /index/status reports any corpus/index
            # divergence live via ``corpus_matches_index``.
        logger.info(
            "Ingested drug label '{}': +{} new chunks ({} replaced).",
            sanitize_log_value(drug_name), chunks_added, chunks_replaced,
        )
        audit.record(
            event_type="ingest",
            endpoint="/documents/drug-label",
            session_hash="anonymous",
            key_id=audit_key_id,
            intent="ingest",
            is_emergency=False,
            red_rule_ids=(),
            abstained=False,
            degradations=(),
            evidence_chunk_ids=(),
            evidence_count=chunks_added,
            policy_version=MedicalSafetyGuard.POLICY_VERSION,
            latency_ms=0.0,
            status_code=200,
        )
        return {
            "drug_name": drug_name,
            "chunks_added": chunks_added,
            "chunks_replaced": chunks_replaced,
            "corpus_size": len(runtime_holder.runtime.documents),
        }

    def _audit_event(
        ctx: AuditContext,
        event_type: str,
        *,
        status_code: int,
        result: dict[str, Any] | None = None,
        session_id: str | None = None,
        red_rule_ids: tuple[str, ...] = (),
        aggregated: int = 0,
    ) -> None:
        """PHI-free audit record: intent/rules/chunk ids/counters only — never
        the query, answer, OCR or history text. Failure is counted, never raised.

        ``aggregated`` carries the suppressed-triage count of an
        ``emergency_flood`` summary row (see the emergency-flood budget above);
        it rides in the existing ``evidence_count`` field — no schema change.
        The endpoint is sanitized (CWE-117): auth failures log the requested
        path, which is client-controlled.
        """
        endpoint = sanitize_log_value(ctx.endpoint)
        is_emergency_event = event_type in ("emergency_triage", "emergency_flood")
        fields: dict[str, Any] = {
            "event_type": event_type,
            "endpoint": endpoint,
            "session_hash": hash_session(session_id) if session_id else "anonymous",
            "key_id": ctx.key_id,
            "intent": "emergency" if is_emergency_event else "",
            "is_emergency": is_emergency_event,
            "abstained": False,
            "evidence_chunk_ids": (),
            "evidence_count": aggregated,
            "policy_version": MedicalSafetyGuard.POLICY_VERSION,
            "latency_ms": ctx.latency_ms(),
            "status_code": status_code,
            "red_rule_ids": red_rule_ids,
        }
        if result is not None:
            sources = result.get("sources") or []
            fields.update({
                "intent": str(result.get("intent", "")),
                "is_emergency": bool(result.get("is_emergency", False)),
                "abstained": bool(result.get("abstained", False)),
                "degradations": tuple(result.get("degradations") or ()),
                "evidence_chunk_ids": tuple(
                    str(s.get("chunk_id")) for s in sources if s.get("chunk_id")
                ),
                "evidence_count": len(sources),
                "policy_version": int(result.get("policy_version") or MedicalSafetyGuard.POLICY_VERSION),
            })
        audit.record(**fields)
        # W5.4 structured audit line (loguru JSON sink when LOG_FORMAT=json):
        # same PHI-free field set as the JSONL record plus the request
        # correlation id from the middleware context. Hashes only — never the
        # query/answer text, never the key itself.
        logger.bind(
            event="audit",
            event_type=event_type,
            endpoint=endpoint,
            status=int(status_code),
            intent=fields.get("intent", ""),
            is_emergency=bool(fields.get("is_emergency", False)),
            abstained=bool(fields.get("abstained", False)),
            evidence_count=int(fields.get("evidence_count", 0)),
            degradations=[str(code) for code in (fields.get("degradations") or ())],
            policy_version=int(fields.get("policy_version") or 0),
            session_hash=str(fields.get("session_hash", "")),
            key_id=ctx.key_id,
            request_id=get_request_id(),
        ).info("audit {}", event_type)
        # W5.6 consultation outcome counters: intent / abstention / per-code
        # degradations, split at the first colon ("rerank:lexical-fallback"
        # => component=rerank, mode=lexical-fallback — no content, no PHI).
        if result is not None:
            metrics.inc("med_rag_consultations_total", labels={"intent": str(result.get("intent", ""))})
            if result.get("abstained"):
                metrics.inc("med_rag_abstentions_total")
            for code in result.get("degradations") or ():
                component, _, mode = str(code).partition(":")
                metrics.inc(
                    "med_rag_degradations_total",
                    labels={"component": component or "unknown", "mode": mode or "unspecified"},
                )

    def _emergency_response(
        query: str,
        provided_id: str | None = None,
        endpoint: str = "/consultations",
        request: Request | None = None,
    ) -> ConsultationResponse | None:
        """Millisecond triage; runs BEFORE session validation, image reads,
        OCR, retrieval and LLM so an unknown session id can never swallow a
        120 red-line warning.

        Recording the turn into an existing session is best-effort and NEVER
        blocking: a same-session request may be mid-consultation (possibly
        hung on a dead LLM upstream) and hold that session's lock, and the
        image endpoint calls this on the event loop — parking there would
        stall every in-flight request including the next 120 warning. So the
        lock is taken with ``blocking=False``; when it is busy the warning is
        still answered, with a fresh session id instead of the contended one.
        A 120 red-line answer is self-contained constant text, so skipping the
        history write costs nothing safety-relevant.

        Flood contract (see the emergency-flood budget above): a short-circuit
        NEVER creates a session entry. An unauthenticated flood of emergency
        requests used to create one session per request and evict real patient
        sessions from the bounded store; now only an EXISTING provided session
        is written (still non-blocking) and everything else is answered with
        an untracked fresh id — the same shape the capacity-exhausted path
        already returned. A follow-up on that id lands on the ordinary
        404 session-expired path, which the UI maps to "start a new session".
        """
        ctx = AuditContext(endpoint=endpoint, key_id=audit_key_id)
        clean_query = query.strip()
        if not clean_query:
            return None
        # Full-text scan: the 28 red-flag rules are linear-time literal patterns, so
        # the worst case is bounded by the pre-parse 11 MiB body limit rather than a
        # scan window (a head+tail window silently missed mid-text emergencies).
        scan_query = clean_query
        # check_emergency_details keeps the matched rule ids for the audit trail
        # (never the matched query text — PHI boundary).
        verdict = MedicalSafetyGuard.check_emergency_details(scan_query)
        if not verdict.is_emergency:
            return None
        warning = MedicalSafetyGuard.check_emergency(scan_query)[1]
        # Emergency answers are self-contained; a known session still gets its
        # turn recorded, an unknown one is issued a fresh id either way.
        answer = warning or "【医疗急救最高警示】检测到急性急危重症，请立即拨打 120 或前往急诊！"
        issued = None
        metrics.inc("med_rag_emergency_total")
        caller_authorized = reads_open or bool(
            admin_token
            and request is not None
            and (bearer := _extract_bearer(request))
            and secrets.compare_digest(bearer.encode("utf-8"), admin_token.encode("utf-8"))
        )
        normalized_id: str | None = None
        if caller_authorized and provided_id is not None:
            try:
                normalized_id = sessions.validate_session_id(provided_id)
            except ValueError:
                normalized_id = None
        if normalized_id is not None:
            existing = sessions.acquire(normalized_id)
            if existing is not None:
                # Non-blocking: see the docstring — a 120 warning must never
                # queue behind a session lock on the event loop.
                lock = sessions.lock_for(normalized_id)
                if lock.acquire(blocking=False):
                    try:
                        existing.record(
                            clean_query[:MAX_QUERY_CHARS], answer, "emergency",
                            policy_version=MedicalSafetyGuard.POLICY_VERSION,
                        )
                    finally:
                        lock.release()
                        sessions.release(normalized_id)
                    issued = normalized_id
                else:
                    # The same session is busy (likely a hung upstream): drop
                    # nothing, just do not record. Fall through to the fresh
                    # (untracked) id below.
                    sessions.release(normalized_id)
        if issued is None:
            # Untracked on purpose: no store entry, no eviction pressure, and
            # no capacity limit can ever swallow the warning.
            issued = sessions.new_session_id()
        _record_emergency_audit(ctx, issued, verdict.rule_ids)
        return ConsultationResponse(
            answer=answer, sources=[], is_emergency=True, intent="emergency", session_id=issued
        )

    def _record_emergency_audit(
        ctx: AuditContext, session_id: str, red_rule_ids: tuple[str, ...]
    ) -> None:
        """Audit one emergency triage under the flood budget.

        Within budget: the individual PHI-free row, exactly as before. Beyond
        it: no per-request row — the flood is counted (metrics + /health) and
        summarized by one aggregate ``emergency_flood`` row per window, written
        when the next triage observes the window rollover. The 120 answer is
        byte-identical either way.
        """
        individual, pending_aggregate = _charge_emergency_flood()
        if pending_aggregate:
            # Chronologically first: it summarizes the window that just ended.
            _audit_event(ctx, "emergency_flood", status_code=200, aggregated=pending_aggregate)
        if individual:
            _audit_event(
                ctx, "emergency_triage", status_code=200,
                session_id=session_id, red_rule_ids=red_rule_ids,
            )

    @app.post("/consultations", response_model=ConsultationResponse)
    def consult(
        payload: ConsultationRequest, request: Request, runtime: AppRuntime = Depends(get_runtime)
    ) -> ConsultationResponse:
        ctx = AuditContext(endpoint="/consultations", key_id=audit_key_id)

        # Invariant #3 + P2-c/F: emergency triage runs BEFORE authentication and
        # BEFORE request-content validation — a missing/invalid key or an
        # over-long pasted history must never delay or swallow a 120 red-line
        # warning (the emergency answer is a constant safety script with no
        # data-leak surface). Ordinary requests pay the validation + auth check
        # here, byte-for-byte identical to the old route-level dependency.
        emergency = _emergency_response(payload.query, payload.session_id, request=request)
        if emergency is not None:
            return emergency

        require_reader(request)
        clean_query = payload.query.strip()
        if not clean_query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        if len(clean_query) > MAX_QUERY_CHARS:
            raise HTTPException(status_code=422, detail=f"query too long (max {MAX_QUERY_CHARS} chars)")

        try:
            requested_id = sessions.validate_session_id(payload.session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _throttle(request, ctx.endpoint)

        with _heavy_slot():
            if requested_id is not None:
                session = sessions.acquire(requested_id)
                if session is None:
                    raise HTTPException(status_code=404, detail="unknown or expired session")
                with sessions.session_lock(requested_id):
                    result = runtime.workflow.run(clean_query, session=session)
                _audit_event(ctx, "consultation", status_code=200, result=result, session_id=requested_id)
                return ConsultationResponse(**result, session_id=requested_id)

            try:
                issued, session = sessions.create()
            except SessionStoreFull as exc:
                _audit_event(ctx, "consultation", status_code=503)
                raise HTTPException(
                    status_code=503, detail="session capacity exhausted; all sessions are active"
                ) from exc
            with sessions.session_lock(issued):
                result = runtime.workflow.run(clean_query, session=session)
            _audit_event(ctx, "consultation", status_code=200, result=result, session_id=issued)
            return ConsultationResponse(**result, session_id=issued)

    @app.post("/consultations/with-image", response_model=ConsultationResponse)
    async def consult_with_image(
        request: Request,
        query: str = Form("请给出该药品的说明书与注意事项"),
        image: UploadFile | None = File(None),
        session_id: str | None = Form(None),
        runtime: AppRuntime = Depends(get_runtime),
    ) -> ConsultationResponse:
        """OCR a drug-box photo, fuzzy-match the drug name, then consult."""
        ctx = AuditContext(endpoint="/consultations/with-image", key_id=audit_key_id)

        # Invariant #3: Emergency triage MUST short-circuit in milliseconds before
        # session validation, reading images, running OCR, or triggering RAG —
        # and now also before authentication and before the required-field /
        # length validation (P2-c/F + review wave-8 B-1): a 120 red-line warning
        # is a constant safety script, so neither auth nor schema-level
        # validation may delay or suppress it.
        emergency = await run_in_threadpool(
            _emergency_response, query, session_id, ctx.endpoint, request
        )
        if emergency is not None:
            return emergency

        await run_in_threadpool(require_reader, request)
        # Validation AFTER triage (schema field constraints on this route are
        # deliberately loosened to let the triage above see malformed requests).
        if image is None:
            raise HTTPException(status_code=422, detail="image is required")
        clean_query = query.strip()
        if not clean_query:
            raise HTTPException(status_code=422, detail="query must not be empty")
        if len(clean_query) > MAX_QUERY_CHARS:
            raise HTTPException(status_code=422, detail=f"query too long (max {MAX_QUERY_CHARS} chars)")

        try:
            requested_id = sessions.validate_session_id(session_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _throttle(request, ctx.endpoint)

        with _heavy_slot():
            return await _consult_with_image_locked(image, clean_query, requested_id, ctx, runtime)

    async def _consult_with_image_locked(
        image: UploadFile,
        clean_query: str,
        requested_id: str | None,
        ctx: AuditContext,
        runtime: AppRuntime,
    ) -> ConsultationResponse:
        """The expensive half of /consultations/with-image: image checks, OCR, RAG, LLM.

        Runs while the caller holds one ``_heavy_slot``.
        """
        import os
        import tempfile

        raw_ext = (image.filename or "").rsplit(".", 1)[-1].lower() if "." in (image.filename or "") else ""
        allowed_exts = {"jpg", "jpeg", "png", "bmp", "webp"}
        if raw_ext and raw_ext not in allowed_exts:
            raise HTTPException(status_code=400, detail=f"unsupported image extension: .{raw_ext}")
        suffix = f".{raw_ext}" if raw_ext in allowed_exts else ".jpg"
        data = await image.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="image too large (max 10 MiB)")
        if not _has_valid_image_magic(data):
            raise HTTPException(status_code=400, detail="file content is not a supported image")
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                tmp.write(data)

            drug_names = sorted(
                {
                    name
                    for doc in runtime.documents
                    if (name := str(doc.metadata.get("drug_name", "")).strip()) and name != "未知药品"
                }
            )
            # Keep blocking OCR/model inference off the event loop. The OCR
            # status rides on the return value (not a shared attribute) so a
            # concurrent with-image request cannot overwrite this request's
            # failure with its own "ok".
            lines, matched, ocr_status = await run_in_threadpool(
                runtime.ocr.suggest_drug_from_image, tmp_path, drug_names
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError as exc:
                    # A locked temp file (AV/EDM scanner) must never mask the
                    # OCR result or turn a succeeded consultation into a 500.
                    logger.warning("Temp image cleanup failed for '{}': {}", tmp_path, exc)

        drug_context = f"OCR 识别文本: {' / '.join(lines[:10])}" if lines else None
        if matched:
            drug_context = f"识别药品: {matched}（置信匹配）" + (f"；OCR 原文: {' / '.join(lines[:10])}" if lines else "")

        async def _respond(result: ConsultationResult, issued_id: str, ocr_status: str) -> ConsultationResponse:
            """Merge OCR status into the response's degradation surface.

            A photo the system failed to read must be distinguishable from an
            ordinary text consultation: the per-response ``ocr_status`` plus a
            ``degradations`` entry mirror /health for the single request.
            Non-mutating: the frozen ConsultationResult is copied via dict()
            instead of the old ``result.pop`` (a frozen Mapping has no pop).
            """
            merged = dict(result)
            degradations = list(merged.get("degradations", []))
            if ocr_status != "ok":
                degradations.append(f"ocr:{ocr_status}")
            merged["degradations"] = degradations
            await run_in_threadpool(
                lambda: _audit_event(
                    ctx, "consultation", status_code=200,
                    result={**merged, "degradations": tuple(degradations)},
                    session_id=issued_id,
                )
            )
            return ConsultationResponse(
                **merged, session_id=issued_id, ocr_status=ocr_status
            )

        def _run_session_locked(session_key: str, session) -> ConsultationResult:
            """Run the workflow under the session lock inside a worker thread.

            ``threading.Lock.acquire`` must never execute on the event loop: a
            second concurrent request for the same session would block the loop
            itself and stall every other request, including emergency triage.
            The whole acquire → run → release critical section therefore runs
            via ``run_in_threadpool``.
            """
            with sessions.session_lock(session_key):
                return runtime.workflow.run(
                    clean_query, drug_context=drug_context, session=session
                )

        if requested_id is not None:
            session = sessions.acquire(requested_id)
            if session is None:
                raise HTTPException(status_code=404, detail="unknown or expired session")
            result = await run_in_threadpool(_run_session_locked, requested_id, session)
            return await _respond(result, requested_id, ocr_status)

        try:
            issued, session = sessions.create()
        except SessionStoreFull as exc:
            await run_in_threadpool(lambda: _audit_event(ctx, "consultation", status_code=503))
            raise HTTPException(
                status_code=503, detail="session capacity exhausted; all sessions are active"
            ) from exc
        result = await run_in_threadpool(_run_session_locked, issued, session)
        return await _respond(result, issued, ocr_status)

    @app.get("/index/status", dependencies=[Depends(require_reader)])
    def index_status(runtime: AppRuntime = Depends(get_runtime)) -> dict[str, Any]:
        manifest = index_manager.current if index_manager is not None else None
        # Report the serving collection from the manager's active binding so the
        # manifest pointer and the serving collection cannot disagree here:
        # the snapshot's dense_encoder lags the manifest for the duration of the
        # post-rebuild serving swap (a full re-embed inside HybridRetriever).
        dense = index_manager.active_encoder if index_manager is not None else None
        if dense is None:
            dense = runtime.dense_encoder
        # Live staleness probe: the manifest pins the corpus hash it was built
        # from, while ingestion moves the authoritative corpus forward without
        # touching collections (rebuild is their only writer). This reports
        # whether the serving corpus still matches the pin — the runtime
        # counterpart of the boot-time /health ``index_stale`` check.
        corpus_matches = None
        if manifest is not None:
            corpus_matches = (
                index_manager.corpus_hash(runtime.documents) == manifest.get("corpus_sha256")
            )
        return {
            "manifest": manifest,
            # "ok" | "absent" | "invalid": a corrupted manifest is a distinct
            # state from no manifest at all (see /health index_manifest);
            # manifest_error carries the parse/schema reason for operators.
            "manifest_status": index_manager.manifest_status if index_manager is not None else None,
            "manifest_error": index_manager.manifest_error if index_manager is not None else None,
            # Boot-time pin health: True only when the bound collection matched
            # the manifest and its live chunk count equals the pinned count.
            "pin_verified": index_manager.pin_verified if index_manager is not None else None,
            "dense_available": dense is not None,
            "active_collection": getattr(dense, "collection_name", None),
            "corpus_matches_index": corpus_matches,
        }

    @app.post("/index/rebuild", dependencies=[Depends(require_admin)])
    def rebuild_index(request: Request) -> dict[str, Any]:
        """Blue-green rebuild of the dense index from the authoritative corpus.

        Builds a fresh collection, validates chunk counts, then atomically
        switches the manifest and the serving retriever. On any failure the
        previous manifest/collection stay active (rollback by construction).

        The whole rebuild + serving swap runs under ``ingest_lock`` so it cannot
        interleave with an ingestion: otherwise a concurrent upsert could mutate
        the authoritative corpus between the rebuild's corpus snapshot and the
        serving swap, wiring a retriever built from new documents to an encoder
        indexed from the old snapshot (manifest/corpus divergence). The region
        additionally holds one concurrency-guard slot (``_heavy_slot`` +
        ``_ingest_region``) so rebuild lock waiters are bounded by the guard
        instead of parking unbounded threads in the shared pool.
        """
        _throttle(request, "/index/rebuild")
        if corpus_store is None or injected_documents:
            raise HTTPException(
                status_code=400, detail="index rebuild requires persistent authoritative corpus"
            )
        if index_manager is None or runtime_holder.runtime.dense_encoder is None:
            raise HTTPException(
                status_code=409, detail="dense index backend unavailable; rebuild not possible"
            )
        with _heavy_slot():
            with _ingest_region():
                serving = runtime_holder.runtime
                try:
                    manifest = index_manager.rebuild()
                except Exception as exc:
                    logger.error("Index rebuild rejected: {}", exc)
                    _audit_event(
                        AuditContext(endpoint="/index/rebuild", key_id=audit_key_id),
                        "index_rebuild", status_code=500,
                    )
                    raise HTTPException(
                        status_code=500, detail="index rebuild failed; previous index kept"
                    ) from exc
                # The index was built from the authoritative corpus snapshot. Serve it
                # with a retriever over that same snapshot, not over a possibly-moved
                # corpus, so corpus and embeddings cannot disagree.
                serving_documents = corpus_store.snapshot
                corpus_hash = index_manager.corpus_hash(serving_documents)
                if manifest.get("corpus_sha256") != corpus_hash:
                    # Should be unreachable under the lock, but fail closed rather than
                    # serve an index whose pinned corpus hash no longer matches.
                    logger.error(
                        "Rebuild manifest corpus_sha256={} does not match serving corpus {}; keeping previous index.",
                        manifest.get("corpus_sha256"), corpus_hash[:12],
                    )
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "index rebuild raced a corpus change; serving swap aborted "
                            "(the rebuilt manifest is on disk — check /index/status)"
                        ),
                    )
                # Build the new serving objects BEFORE committing: commit_serving then
                # only performs the workflow.retriever reference write (the commit
                # point of the serving switch) and the single holder swap.
                new_encoder = index_manager.active_encoder
                new_retriever = HybridRetriever(
                    documents=serving_documents, k=config.rrf_k, dense_encoder=new_encoder
                )
                commit_serving(
                    runtime_holder,
                    documents=serving_documents,
                    retriever=new_retriever,
                    dense_encoder=new_encoder,
                    ocr=serving.ocr,
                    workflow=serving.workflow,
                    audit=serving.audit,
                )
        _audit_event(
            AuditContext(endpoint="/index/rebuild", key_id=audit_key_id),
            "index_rebuild", status_code=200,
        )
        return {
            "rebuilt": True,
            "manifest": manifest,
            "dense_degraded": retriever_degraded(runtime_holder.runtime),
        }

    return app


app = create_app()
