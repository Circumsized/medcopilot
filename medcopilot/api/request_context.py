"""Request-context middleware (W5.4): X-Request-ID + structured-log context.

Zero third-party dependencies (ASGI wire format only), so the middleware can
sit OUTSIDE the body-size limiter and hand EVERY response an ``X-Request-ID``
header — including the 401/413/429/503 rejections the middleware stack and the
abuse controls themselves emit. Correlation must survive a rejection: a
throttled or unauthorized caller still needs an id to quote when they call
the clinic, not the server.

Two propagation surfaces, by design:

* ``request_id_var`` — a ``contextvars.ContextVar`` any sink can read
  (audit emission, access logs). Starlette runs each request in its own
  context and ``run_in_threadpool`` copies it into worker threads, so the
  value cannot leak across requests.
* ``logger.contextualize(request_id=...)`` — loguru extra for the structured
  JSON sink (``LOG_FORMAT=json`` in medcopilot/common/logger.py patches
  ``record["extra"]["request_id"]``). Kept as the ONLY channel the log layer
  reads; this module never imports the logger's configuration.

PHI boundary: the request id is an opaque correlation token, never content.
An inbound id is accepted only when it is ≤128 printable-ASCII characters
with no whitespace/control bytes; anything else (empty, over-long, forged,
non-ASCII) is replaced by a server-generated value, so a client cannot inject
log-forging bytes or an id that collides with another caller's.
"""

from __future__ import annotations

import secrets
import sys
import threading
import time
from contextvars import ContextVar
from typing import Any

from medcopilot.common.logger import logger

REQUEST_ID_HEADER = "x-request-id"
REQUEST_ID_MAX_CHARS = 128
#: Fallback id used when no context is set (logs emitted outside a request).
UNKNOWN_REQUEST_ID = "-"
#: Metrics label for any path that is not a registered route. A scanner
#: probing random paths must not grow the ``endpoint`` label cardinality
#: without limit: the registry's MAX_SERIES bound would then silently drop
#: legitimate series once a flood of distinct paths fills the table.
OTHER_ENDPOINT_LABEL = "other"

request_id_var: ContextVar[str] = ContextVar("request_id", default=UNKNOWN_REQUEST_ID)


def generate_request_id() -> str:
    """Opaque, unguessable, url-safe correlation id (128 bits)."""
    return secrets.token_urlsafe(16)


def sanitize_log_value(value: str, limit: int = 200) -> str:
    """CWE-117: make untrusted text safe for a single-line log record.

    换行/制表符转义为字面 ``\\n``/``\\t``，其余非打印字符转 ``\\xNN``，
    超长截断——伪造多行日志条目或注入控制序列的输入在落盘前被中和。
    Lives here (not in server.py) because the access log and the audit
    emitter in server.py both funnel through it.
    """
    cleaned_chars: list[str] = []
    for ch in (value or ""):
        if ch == "\n":
            cleaned_chars.append("\\n")
        elif ch == "\r":
            cleaned_chars.append("\\r")
        elif ch == "\t":
            cleaned_chars.append("\\t")
        elif ch.isprintable():
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(f"\\x{ord(ch):02x}")
    return "".join(cleaned_chars)[:limit]


def is_valid_request_id(raw: bytes | str | None) -> bool:
    """Bounded, printable-ASCII only — no whitespace, no control bytes.

    宽松一点的可打印 ASCII 边界（0x21-0x7E，不含空格）确保入站 id 既不能
    注入日志伪造用的换行/控制序列，也不会把任意 Unicode 塞进 header。
    """
    if not raw:
        return False
    if isinstance(raw, str):
        raw = raw.encode("latin-1", errors="strict") if raw.isascii() else b""
        if not raw:
            return False
    if len(raw) > REQUEST_ID_MAX_CHARS:
        return False
    return all(0x21 <= byte <= 0x7E for byte in raw)


def get_request_id() -> str:
    """Current request's correlation id (``-`` outside a request)."""
    return request_id_var.get()


class RequestContextMiddleware:
    """ASGI middleware: generate/propagate ``X-Request-ID`` + log context.

    Optionally feeds the request counter/duration histogram of a metrics
    registry (duck-typed; ``api/metrics.py``) — the correlation id itself and
    any label used are non-sensitive (endpoint path + numeric status only).
    """

    def __init__(self, app: Any, header: str = REQUEST_ID_HEADER, metrics: Any = None) -> None:
        self.app = app
        self.header = header
        self.metrics = metrics
        self._header_bytes = header.lower().encode("latin-1")
        # Lazily-built set of registered route paths (see _endpoint_label).
        # Double-checked locking: the middleware instance is shared by every
        # request, so the cache write must not race with a concurrent reader.
        self._known_paths: frozenset[str] | None = None
        self._known_paths_lock = threading.Lock()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        inbound: str | None = None
        for key, value in scope.get("headers") or ():
            if key == self._header_bytes and is_valid_request_id(value):
                inbound = value.decode("latin-1")
                break
        request_id = inbound or generate_request_id()

        token = request_id_var.set(request_id)
        started = time.perf_counter()
        observed: dict[str, int] = {}

        async def send_with_request_id(message: dict) -> None:
            if message.get("type") == "http.response.start":
                observed["status"] = int(message.get("status", 0))
                # Append (never replace) so an application-set header wins and
                # duplicate ids cannot be injected by a downstream handler.
                message.setdefault("headers", []).append(
                    (self._header_bytes, request_id.encode("latin-1"))
                )
            await send(message)

        try:
            # loguru contextualize puts request_id into record["extra"] for the
            # duration of the request AND the finally-block access log below.
            with logger.contextualize(request_id=request_id):
                try:
                    await self.app(scope, receive, send_with_request_id)
                finally:
                    self._observe(scope, observed.get("status", 0), time.perf_counter() - started)
        finally:
            request_id_var.reset(token)

    def _observe(self, scope: Any, status: int, seconds: float) -> None:
        self._log_access(scope, status, seconds)
        if self.metrics is None:
            return
        try:
            endpoint = self._endpoint_label(scope)
            self.metrics.inc(
                "med_rag_requests_total",
                labels={"endpoint": endpoint, "status": str(status)},
            )
            self.metrics.observe(
                "med_rag_request_duration_seconds", seconds, labels={"endpoint": endpoint}
            )
        except Exception:  # pragma: no cover - metrics must never break a request
            logger.warning(
                "metrics instrumentation failed for {}",
                sanitize_log_value(str(scope.get("path") or "-")),
            )

    def _endpoint_label(self, scope: Any) -> str:
        """Registered route path, or the single ``other`` bucket.

        ``scope["path"]`` is client-controlled (percent-decoded), so labeling
        by it directly lets any caller mint new series — a scanner walking
        random paths would fill the registry's MAX_SERIES table and silently
        evict legitimate series from the export. Starlette sets
        ``scope["app"]`` before the middleware stack, so the registered route
        set is always available on the request path; without it (raw ASGI
        harness) the path cannot be verified and is bucketed as ``other``.
        """
        path = str(scope.get("path") or "-")
        known = self._registered_paths(scope)
        return path if known is not None and path in known else OTHER_ENDPOINT_LABEL

    def _registered_paths(self, scope: Any) -> frozenset[str] | None:
        app = scope.get("app")
        if app is None:
            return None
        cached = self._known_paths
        if cached is None:
            with self._known_paths_lock:
                if self._known_paths is None:
                    self._known_paths = frozenset(
                        route.path
                        for route in getattr(app, "routes", ())
                        if isinstance(getattr(route, "path", None), str)
                    )
                cached = self._known_paths
        return cached

    def _log_access(self, scope: Any, status: int, seconds: float) -> None:
        """One PHI-free structured access line per request (W5.4).

        Field set matches the ops-design spec: method / endpoint / status /
        duration_ms; the correlation id rides on loguru's ``extra`` via
        ``contextualize`` (the common JSON sink serializes it). No query text,
        no host, no client address, no key material — those stay out.
        Method and endpoint are sanitized (CWE-117): the request path is
        client-controlled and percent-decoding can deliver control bytes
        (e.g. ``%0A``) into a log line.
        """
        try:
            method = sanitize_log_value(str(scope.get("method") or "-"))
            endpoint = sanitize_log_value(str(scope.get("path") or "-"))
            logger.bind(
                event="access",
                method=method,
                endpoint=endpoint,
                status=int(status),
                duration_ms=round(seconds * 1000.0, 3),
                request_id=get_request_id(),
            ).info("access {} {}", method, endpoint)
        except Exception as exc:  # pragma: no cover - logging must never break a request
            try:
                if sys.stderr is not None:
                    print(
                        f"WARNING: access log emission failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
            except Exception:
                pass
