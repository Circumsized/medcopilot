"""Wave 10 (review wave-10): API / audit / observability hardening contract.

Pins the items implemented after the 2026-10-09 expert review (offline, no
model downloads). Deploy/container items live in tests/test_deploy_contract.py
(TestContainerEntrypoint / wrapper scrub / broken symlink).

1. Emergency flood budget (P2 安全/运维): the pre-auth emergency short-circuit
   is unauthenticated and unthrottled by contract, so a flood must not create
   session entries (evicting real patients) nor one audit row per request.
   Answers stay the fixed 120 text; individual audit rows stop at the ordinary
   consultation quota (an existing, operator-tuned budget); the rest is counted
   and summarized by one aggregate row per window.
2. Audit sink fail-closed (P2): a write failure disables the sink, never
   blind-retries (a partial write would fork the hash chain), and never flips
   back to ready on its own.
3. Request context (P2 测试缺口): X-Request-ID validation/propagation, per-
   request contextvar isolation under concurrency, access-log endpoint
   sanitization (CWE-117).
4. Metrics (P2 测试缺口): registry units (cardinality bound, label escaping,
   monotonic set_counter, histogram shape) + unknown-path endpoint label
   normalization (bounded label cardinality).
5. Rate-limit header semantics: X-RateLimit-Reset is the ABSOLUTE epoch second
   while Retry-After stays a relative duration; the limiter's hard key bound
   fails closed instead of growing the table.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from medcopilot.api.metrics import MAX_SERIES, MetricsRegistry
from medcopilot.api.rate_limit import RateLimitExceeded, SlidingWindowLimiter
from medcopilot.api.request_context import (
    REQUEST_ID_MAX_CHARS,
    RequestContextMiddleware,
    get_request_id,
    is_valid_request_id,
)
from medcopilot.audit import AuditLogger
from medcopilot.audit import audit_log as audit_log_module
from medcopilot.common.config import config as app_config

EMERGENCY_QUERY = "突发心肌梗死，胸痛压榨濒死感"
PLAIN_QUERY = "头痛如何治疗？"
LABEL_A = "【药品名称】药品甲\n【适应症】用于甲。\n【用法用量】口服。一次1片，一日3次。\n【禁忌】过敏者禁用。\n"
LABEL_B = "【药品名称】药品乙\n【适应症】用于乙。\n【用法用量】口服。一次2片，一日2次。\n【禁忌】孕妇禁用。\n"


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _fields(**overrides):
    base = {
        "event_type": "consultation",
        "endpoint": "/consultations",
        "session_hash": "anonymous",
        "key_id": "k" * 16,
        "intent": "medical",
        "is_emergency": False,
        "abstained": False,
        "evidence_chunk_ids": (),
        "evidence_count": 0,
        "policy_version": 2,
        "latency_ms": 1.0,
        "status_code": 200,
    }
    base.update(overrides)
    return base


def _isolated_app(monkeypatch, tmp_path, *, clock=None, encoder_factory=None, **overrides):
    """An isolated dev app: own audit file, own config copy (no test leakage).

    The config copy is taken from the real singleton, so a test may build
    several generations of the app on the same tmp corpus (monkeypatch
    restores the module attribute afterwards).
    """
    from fastapi.testclient import TestClient

    import medcopilot.api.server as server_module

    configured = SimpleNamespace(**app_config.model_dump())
    configured.audit_log_path = str(tmp_path / "audit.jsonl")
    configured.project_env = "development"
    configured.admin_api_key = None
    for name, value in overrides.items():
        setattr(configured, name, value)
    monkeypatch.setattr(server_module, "config", configured)
    factory_kwargs: dict = {}
    if clock is not None:
        factory_kwargs["clock"] = clock
    if encoder_factory is not None:
        factory_kwargs["dense_encoder_factory"] = encoder_factory
    app = server_module.create_app(**factory_kwargs)
    return TestClient(app), app, tmp_path / "audit.jsonl"


def _flood_app(monkeypatch, tmp_path, *, consult_per_minute: int, clock=None):
    client, _app, audit_path = _isolated_app(
        monkeypatch,
        tmp_path,
        clock=clock,
        rate_limit_enabled=True,
        rate_limit_consult_per_minute=consult_per_minute,
    )
    return client, audit_path


def _events(audit_path) -> list[dict]:
    return [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]


# ------------------------------------------- 1: emergency flood budget


def test_emergency_flood_keeps_fixed_text_and_creates_no_sessions(monkeypatch, tmp_path):
    client, audit_path = _flood_app(monkeypatch, tmp_path, consult_per_minute=2)
    answers = []
    for _ in range(8):
        response = client.post("/consultations", json={"query": EMERGENCY_QUERY})
        assert response.status_code == 200
        body = response.json()
        assert body["is_emergency"] is True and "120" in body["answer"]
        answers.append(body["answer"])
    assert len(set(answers)) == 1, "over-budget answers must stay the same fixed 120 text"

    health = client.get("/health").json()
    # A flood must not mint session entries (the old path evicted real ones).
    assert health["sessions_active"] == 0
    assert health["components"]["abuse_controls"]["emergency_flood_suppressed"] == 6

    triages = [e for e in _events(audit_path) if e["event_type"] == "emergency_triage"]
    assert len(triages) == 2, "only the in-budget requests may get an individual audit row"


def test_emergency_flood_aggregate_row_is_written_once_per_window(monkeypatch, tmp_path):
    clock = FakeClock()
    client, audit_path = _flood_app(monkeypatch, tmp_path, consult_per_minute=2, clock=clock)
    for _ in range(4):  # 2 individual + 2 suppressed in window #1
        assert client.post("/consultations", json={"query": EMERGENCY_QUERY}).status_code == 200
    clock.now += 61.0  # the consultation window rolls over
    assert client.post("/consultations", json={"query": EMERGENCY_QUERY}).status_code == 200

    floods = [e for e in _events(audit_path) if e["event_type"] == "emergency_flood"]
    assert len(floods) == 1, "one aggregate summary row per elapsed window"
    assert floods[0]["evidence_count"] == 2  # the suppressed triages it summarizes
    assert floods[0]["is_emergency"] is True and floods[0]["intent"] == "emergency"
    assert floods[0]["status_code"] == 200


def test_emergency_audit_budget_follows_the_consultation_quota(monkeypatch, tmp_path):
    """The budget is the existing operator-tuned quota — no new magic number."""
    client, audit_path = _flood_app(monkeypatch, tmp_path, consult_per_minute=1)
    for _ in range(2):
        assert client.post("/consultations", json={"query": EMERGENCY_QUERY}).status_code == 200
    assert len([e for e in _events(audit_path) if e["event_type"] == "emergency_triage"]) == 1
    suppressed = client.get("/health").json()["components"]["abuse_controls"]
    assert suppressed["emergency_flood_suppressed"] == 1


def test_emergency_answers_even_when_the_ordinary_quota_is_saturated(monkeypatch, tmp_path):
    client, _ = _flood_app(monkeypatch, tmp_path, consult_per_minute=1)
    session_id = client.post("/consultations", json={"query": PLAIN_QUERY}).json()["session_id"]
    assert client.post("/consultations", json={"query": PLAIN_QUERY}).status_code == 429
    # The limiter is full, yet the red-line answer still returns and reuses
    # the known session (its turn is still recorded).
    emergency = client.post("/consultations", json={"query": EMERGENCY_QUERY, "session_id": session_id})
    assert emergency.status_code == 200
    assert emergency.json()["is_emergency"] is True
    assert emergency.json()["session_id"] == session_id
    for _ in range(5):
        flooded = client.post("/consultations", json={"query": EMERGENCY_QUERY})
        assert flooded.status_code == 200 and "120" in flooded.json()["answer"]


def test_known_session_history_survives_an_emergency_turn(monkeypatch, tmp_path):
    client, _ = _flood_app(monkeypatch, tmp_path, consult_per_minute=5)
    session_id = client.post("/consultations", json={"query": PLAIN_QUERY}).json()["session_id"]
    emergency = client.post("/consultations", json={"query": EMERGENCY_QUERY, "session_id": session_id})
    assert emergency.status_code == 200 and emergency.json()["session_id"] == session_id
    follow_up = client.post("/consultations", json={"query": "后续咨询", "session_id": session_id})
    assert follow_up.status_code == 200

    # The emergency turn is recorded INTO the live session (not dropped, not
    # replaced by an untracked id): the pre-emergency history is still there.
    store = client.app.state.sessions
    session = store.acquire(session_id)
    try:
        turns = [(turn["query"], turn["intent"]) for turn in session.turns]
    finally:
        store.release(session_id)
    assert (PLAIN_QUERY, "medical") in turns
    assert (EMERGENCY_QUERY, "emergency") in turns


# --------------------------------------------- 2: audit fail-closed


def test_write_failure_disables_sink_without_retry_or_chain_fork(tmp_path, monkeypatch):
    """A failed append must not be retried: the record may have partially
    landed, and a second append would chain onto a stale head (fork)."""
    path = tmp_path / "audit.jsonl"
    audit = AuditLogger(str(path), fsync=True)
    assert audit.record(**_fields()) is not None
    head_before = audit.chain_head()

    def failing_fsync(_fd: int) -> None:
        raise OSError("fsync failed")

    error_logs: list[tuple[str, tuple]] = []

    def recording_error(message: str, *args) -> None:
        error_logs.append((message, args))

    monkeypatch.setattr(audit_log_module.os, "fsync", failing_fsync)
    monkeypatch.setattr(audit_log_module.logger, "error", recording_error)

    assert audit.record(**_fields()) is None
    assert audit.ready is False and audit.failures == 1
    # The disabled sink is loud: exactly one error line per failed attempt,
    # naming the failure and the fail-closed state (no silent drop).
    assert len(error_logs) == 1, "the disabled sink must log the failure exactly once"
    message, args = error_logs[0]
    assert "sink disabled until recovery" in message
    assert args == ("OSError",)
    assert audit.chain_head() == head_before, "the in-memory chain head must not advance on failure"

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, "the failing record landed once; a blind retry would append it twice"
    ok, broken = AuditLogger(str(path)).verify_chain()
    assert ok is True and broken == 0, "the on-disk chain must stay verifiable (no fork)"

    # No recovery without verification: the sink stays disabled for the process.
    # Refused attempts are counted but NOT re-logged — logging every refused
    # record would turn a flood against a disabled sink into a log amplifier.
    assert audit.record(**_fields()) is None
    assert audit.ready is False and audit.failures == 2
    assert len(error_logs) == 1


def test_unwritable_sink_counts_every_attempt_and_stays_disabled(tmp_path):
    blocked = tmp_path / "audit.jsonl"
    blocked.mkdir()  # open(..., "a") raises IsADirectoryError (an OSError)
    audit = AuditLogger(str(blocked))
    for _ in range(3):
        assert audit.record(**_fields()) is None
    assert audit.failures == 3 and audit.ready is False
    assert audit.chain_head() == "GENESIS"


def test_failed_audit_sink_is_visible_on_health(monkeypatch, tmp_path):
    """fail-closed contract: a sink that cannot write must surface on /health
    (audit_ready=false, failures>0) instead of silently dropping records."""
    # The app's own audit path, occupied by a directory: every append raises
    # IsADirectoryError (an OSError), exactly like a full/removed volume.
    (tmp_path / "audit.jsonl").mkdir()
    client, _app, _audit = _isolated_app(monkeypatch, tmp_path)

    # The request itself still succeeds — an audit failure never breaks the
    # consultation path (fail-safe); it only stops being invisible.
    assert client.post("/consultations", json={"query": PLAIN_QUERY}).status_code == 200

    health = client.get("/health").json()
    assert health["audit_ready"] is False
    assert health["audit_failures"] >= 1
    assert health["components"]["audit"]["ready"] is False
    assert health["components"]["audit"]["failures"] >= 1
    assert health["audit_sink"] is not None


# --------------------------------------------- 3: request context


@pytest.mark.parametrize(
    "raw",
    [
        "abcDEF0123-_",
        "x" * REQUEST_ID_MAX_CHARS,
        "fixed-sample-request-id",
    ],
)
def test_valid_request_ids_are_accepted(raw):
    assert is_valid_request_id(raw) is True
    assert is_valid_request_id(raw.encode("latin-1")) is True


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "has space",
        "has\ttab",
        "has\nnewline",
        "has\x01control",
        "ünïcøde-id",
        "x" * (REQUEST_ID_MAX_CHARS + 1),
    ],
)
def test_invalid_request_ids_are_rejected(raw):
    assert is_valid_request_id(raw) is False


def test_request_id_propagation_and_replacement(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import medcopilot.api.server as server_module

    app = server_module.create_app()
    client = TestClient(app)

    inbound = "correlation-id-from-caller"
    echoed = client.get("/health", headers={"x-request-id": inbound})
    assert echoed.headers["x-request-id"] == inbound

    # Forged/oversized/whitespace ids never reach the response or the logs
    # (a newline in an echoed header would be response-header injection).
    # Non-ASCII ids stay in the unit test above: httpx cannot even put them
    # on the wire, but a raw socket client could.
    for hostile in ("forged\ninjected-line", "x" * 500, "sp ace", "tab\there"):
        replaced = client.get("/health", headers={"x-request-id": hostile})
        assert replaced.headers["x-request-id"] != hostile
        assert is_valid_request_id(replaced.headers["x-request-id"])

    # Even a rejection (404) carries a correlation id — the middleware is the
    # outermost layer, so a throttled/unknown-path caller can quote it.
    missing = client.get("/definitely-not-a-route")
    assert missing.status_code == 404
    assert is_valid_request_id(missing.headers["x-request-id"])


def test_concurrent_requests_do_not_leak_request_ids():
    """The id lives in a contextvar: two in-flight requests must each observe
    their own value even when their handlers interleave."""
    seen: list[tuple[str, str]] = []

    async def inner(scope, receive, send):
        inbound = dict(scope["headers"]).get(b"x-request-id", b"?").decode("latin-1")
        await asyncio.sleep(0.01)  # widen the window a global variable would lose
        seen.append((inbound, get_request_id()))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = RequestContextMiddleware(inner)

    def scope_for(request_id: str) -> dict:
        return {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/health",
            "raw_path": b"/health",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"testserver"), (b"x-request-id", request_id.encode("latin-1"))],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def drive(request_id: str) -> None:
        await middleware(scope_for(request_id), receive, _noop_send)

    async def _noop_send(_message: dict) -> None:
        return None

    async def _drive_both() -> None:
        await asyncio.gather(drive("req-alpha-111"), drive("req-bravo-222"))

    asyncio.run(_drive_both())
    assert sorted(seen) == [("req-alpha-111", "req-alpha-111"), ("req-bravo-222", "req-bravo-222")]
    # The contextvar is reset after each request — nothing leaks to the next one.
    assert get_request_id() == "-"


def test_access_log_endpoint_is_sanitized():
    """A percent-decoded control byte in the path must not forge a log line."""
    from medcopilot.common.logger import logger

    records: list[str] = []
    handle = logger.add(lambda message: records.append(str(message)), level="INFO", format="{message}")
    try:
        middleware = RequestContextMiddleware(_unused_app)
        middleware._log_access(
            {"method": "GET", "path": "/health\nforged-access-line\t2026-01-01"}, 200, 0.5
        )
    finally:
        logger.remove(handle)
    assert records, "an access line must have been emitted"
    line = records[-1]
    # loguru's sink terminates the record with exactly one newline; the
    # payload itself must contribute none.
    assert line.count("\n") == 1 and line.count("\t") == 0
    assert "\\n" in line and "\\t" in line
    assert "forged-access-line" in line  # content preserved, control bytes escaped


async def _unused_app(scope, receive, send):  # pragma: no cover - placeholder
    return None


# --------------------------------------------- 4: metrics


def test_registry_drops_new_series_past_the_cardinality_bound():
    registry = MetricsRegistry()
    for index in range(MAX_SERIES):
        registry.inc("med_rag_probe", labels={"k": str(index)})
    assert registry.dropped_series == 0
    registry.inc("med_rag_probe", labels={"k": "one-too-many"})
    assert registry.dropped_series == 1
    # Existing series keep accumulating; nothing is silently evicted.
    registry.inc("med_rag_probe", labels={"k": "0"})
    assert 'med_rag_probe{k="0"} 2' in registry.export_prom()
    assert 'med_rag_dropped_series_total 1' in registry.export_prom()


def test_registry_escapes_label_values_and_formats_histograms():
    registry = MetricsRegistry()
    registry.inc("med_rag_quoted", labels={"endpoint": 'a"b\\c\nd'})
    registry.observe("med_rag_seconds", 0.2, labels={"endpoint": "/health"})
    text = registry.export_prom()
    assert 'med_rag_quoted{endpoint="a\\"b\\\\c\\nd"}' in text
    assert "# TYPE med_rag_seconds histogram" in text
    assert 'med_rag_seconds_bucket{endpoint="/health",le="0.25"} 1' in text
    assert 'med_rag_seconds_bucket{endpoint="/health",le="+Inf"} 1' in text
    assert 'med_rag_seconds_sum{endpoint="/health"} 0.2' in text
    assert 'med_rag_seconds_count{endpoint="/health"} 1' in text


def test_set_counter_never_decreases():
    registry = MetricsRegistry()
    registry.set_counter("med_rag_audit_write_failures_total", 5)
    registry.set_counter("med_rag_audit_write_failures_total", 2)
    assert "med_rag_audit_write_failures_total 5" in registry.export_prom()


def test_unknown_paths_are_bucketed_into_one_metrics_label(monkeypatch, tmp_path):
    """A scanner walking random paths must not mint new label series."""
    client, app, _audit = _isolated_app(monkeypatch, tmp_path)

    for index in range(20):
        assert client.get(f"/scanner-probe-{index}").status_code == 404
    assert client.get("/health").status_code == 200

    registry = app.state.metrics
    endpoints = {
        line.split('endpoint="')[1].split('"')[0]
        for line in registry.export_prom().splitlines()
        if line.startswith("med_rag_requests_total{")
    }
    assert endpoints == {"/health", "other"}, endpoints
    assert registry.dropped_series == 0


# --------------------------------------- 5: rate-limit header semantics


def test_rate_limit_reset_header_is_absolute_epoch(monkeypatch, tmp_path):
    client, _ = _flood_app(monkeypatch, tmp_path, consult_per_minute=1)
    assert client.post("/consultations", json={"query": PLAIN_QUERY}).status_code == 200
    blocked = client.post("/consultations", json={"query": PLAIN_QUERY})
    assert blocked.status_code == 429

    retry_after = int(blocked.headers["Retry-After"])
    reset = int(blocked.headers["X-RateLimit-Reset"])
    now = int(time.time())
    assert 1 <= retry_after <= 60
    # Absolute epoch second (IETF ratelimit-headers), not a relative delta.
    assert now + retry_after - 2 <= reset <= now + retry_after + 2
    assert blocked.headers["X-RateLimit-Limit"] == "1"
    assert blocked.headers["X-RateLimit-Remaining"] == "0"
    assert blocked.json()["detail"]["retry_after_seconds"] == retry_after


def test_rate_limit_reset_header_is_exposed_to_browsers(monkeypatch, tmp_path):
    client, _app, _audit = _isolated_app(monkeypatch, tmp_path, cors_allow_all=True)

    exposed = client.get("/health", headers={"Origin": "http://localhost:7860"}).headers.get(
        "access-control-expose-headers", ""
    )
    assert "X-RateLimit-Reset" in exposed


def test_limiter_hard_bound_fails_closed_when_the_key_table_is_full():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, 60.0, clock=clock)
    limiter.MAX_KEYS = 3
    for name in "abc":
        limiter.check(name)
    with pytest.raises(RateLimitExceeded):
        limiter.check("unseen-key")  # every slot held by an in-window identity
    assert limiter.rejected == 1
    clock.now += 61.0  # the sweep can now reclaim the drained keys
    limiter.check("unseen-key")
    assert set(limiter._events) == {"unseen-key"}


# ------------------------------ 6: live index staleness (wave-10)


class _VerifiedStubEncoder:
    """Stub dense encoder that exposes a full embedding identity.

    ``IndexManager`` reports a verified pin only when model/function/dimension
    are all observable, so a bare test double would leave ``pin_verified``
    False after a rebuild and could not pin the "both recover" direction.
    """

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name
        self.docs: list = []
        self.embedding_function = SimpleNamespace(
            model_name="stub-embed-model", name="stub-embed-fn"
        )
        self.embedding_dim = 384

    def measure_dim(self) -> int:
        return 384

    def index_documents(self, documents) -> int:
        self.docs = list(documents)
        return len(self.docs)

    def count(self) -> int:
        return len(self.docs)

    def indexed_ids(self):
        return [str(doc.metadata.get("chunk_id")) for doc in self.docs]

    def __call__(self, text):
        if isinstance(text, list):
            raise TypeError("per-document encoder only")
        return [0.0, 0.0, 0.0]


def _index_app(monkeypatch, tmp_path):
    return _isolated_app(
        monkeypatch,
        tmp_path,
        med_corpus_path=str(tmp_path / "corpus.jsonl"),
        enable_chroma_dense=True,
        encoder_factory=_VerifiedStubEncoder,
    )


def test_health_index_stale_tracks_ingestion_like_index_status(monkeypatch, tmp_path):
    """A post-rebuild ingestion must stale BOTH surfaces.

    /health used to report the boot-time constant while /index/status computed
    the fact live, so a corpus that moved past the pin stayed invisible on the
    readiness surface (a fake green).
    """
    client, _app, _audit = _index_app(monkeypatch, tmp_path)
    assert client.post(
        "/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "药品甲"}
    ).status_code == 200
    assert client.post("/index/rebuild").status_code == 200
    # Baseline: the manifest pins the corpus as ingested so far.
    assert client.get("/index/status").json()["corpus_matches_index"] is True
    assert client.get("/health").json()["index_stale"] is False

    client.post("/documents/drug-label", data={"raw_text": LABEL_B, "drug_name": "药品乙"})

    status = client.get("/index/status").json()
    health = client.get("/health").json()
    assert status["corpus_matches_index"] is False
    assert health["index_stale"] is True, "/health must not keep the boot-time value"
    assert health["components"]["index"]["stale"] is True
    assert health["ready"] is False, "a stale dense index is not ready (no fake green)"


def test_health_index_stale_recovers_after_rebuild(monkeypatch, tmp_path):
    """Boot finds the corpus past the pin (crash between the durable corpus
    commit and the Chroma mirror); a successful rebuild must clear /health's
    staleness — the boot-time constant never recovered from this state."""
    # Generation 1: pin the corpus, then move it past the manifest and abandon
    # the app (the crash window).
    client, _app, _audit = _index_app(monkeypatch, tmp_path)
    client.post("/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "药品甲"})
    assert client.post("/index/rebuild").status_code == 200
    client.post("/documents/drug-label", data={"raw_text": LABEL_B, "drug_name": "药品乙"})

    # Generation 2 boots on the moved corpus: the divergence is visible.
    client2, _app2, _audit2 = _index_app(monkeypatch, tmp_path)
    boot = client2.get("/health").json()
    assert boot["index_stale"] is True and boot["ready"] is False

    assert client2.post("/index/rebuild").status_code == 200
    health = client2.get("/health").json()
    status = client2.get("/index/status").json()
    assert status["corpus_matches_index"] is True and status["pin_verified"] is True
    assert health["index_stale"] is False and health["components"]["index"]["stale"] is False
    assert health["ready"] is True, "a verified, corpus-matching rebuild must be ready again"


def test_index_stale_gauge_follows_the_live_value(monkeypatch, tmp_path):
    """/metrics must report the same live staleness as /health."""
    client, _app, _audit = _index_app(monkeypatch, tmp_path)
    client.post("/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "药品甲"})
    assert client.post("/index/rebuild").status_code == 200
    client.post("/documents/drug-label", data={"raw_text": LABEL_B, "drug_name": "药品乙"})

    # Dev no-key mode opens the admin endpoints, so the scrape needs no bearer.
    body = client.get("/metrics").text
    assert "med_rag_index_stale 1" in body
    assert client.get("/health").json()["index_stale"] is True


def test_metrics_scrape_requires_configured_admin_bearer(monkeypatch, tmp_path):
    client, _, _ = _isolated_app(monkeypatch, tmp_path, admin_api_key="metrics-contract-key")
    assert client.get("/metrics").status_code == 401
    response = client.get("/metrics", headers={"Authorization": "Bearer metrics-contract-key"})
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "med_rag_index_stale" in response.text


def test_index_staleness_cache_does_not_compare_runtime_values(monkeypatch, tmp_path):
    from dataclasses import replace

    from medcopilot.api.runtime import AppRuntime

    client, app, _ = _index_app(monkeypatch, tmp_path)
    client.post("/documents/drug-label", data={"raw_text": LABEL_A, "drug_name": "药品甲"})
    assert client.post("/index/rebuild").status_code == 200
    assert client.get("/health").json()["index_stale"] is False

    def forbidden_value_comparison(self, other):
        raise AssertionError("runtime identity cache must not invoke dataclass value equality")

    monkeypatch.setattr(AppRuntime, "__eq__", forbidden_value_comparison)
    holder = app.state.runtime_holder
    holder.swap(replace(holder.runtime, documents=list(holder.runtime.documents)))
    assert client.get("/health").json()["index_stale"] is False
