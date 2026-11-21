"""W2.3 API rate limiting + expensive-path concurrency guard contract.

OWASP API4:2023 (unrestricted resource consumption) / LLM10:2025 (unbounded
consumption). Three layers, each pinned where it lives:

1. ``SlidingWindowLimiter`` / ``ConcurrencyGuard`` units: exact window maths,
   per-identity isolation, bounded key table, slot release on every exit path.
2. Quota contract over HTTP: 429 + Retry-After / X-RateLimit-* + the 120
   notice, per-endpoint counters, emergency triage exempt, auth failures and
   413s do not burn the caller's quota, limiter off => never 429.
3. Concurrency: when every slot is held by a stuck LLM call the overflow is
   rejected immediately with 503 (never parked) while emergency triage and
   /health stay live.
"""

from __future__ import annotations

import threading
import time

import pytest

from medcopilot.api.rate_limit import (
    ConcurrencyGuard,
    ConcurrencyLimitExceeded,
    RateLimitExceeded,
    SlidingWindowLimiter,
)

EMERGENCY_QUERY = "突发心肌梗死怎么办"
PLAIN_QUERY = "头痛如何治疗？"
ADMIN_KEY = "secret-key"
AUTH = {"Authorization": f"Bearer {ADMIN_KEY}"}
JPEG = b"\xff\xd8\xff" + b"rest-of-image"
LABEL = (
    "【药品名称】阿莫西林胶囊\n"
    "【适应症】适用于敏感菌引起的感染。\n"
    "【用法用量】口服。成人一次0.5g，一日3次。\n"
    "【禁忌】青霉素过敏者禁用。" + "——" * 20
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MOCK_LLM", "1")


# ------------------------------------------------------------ units: limiter


def test_limiter_blocks_past_limit_and_reports_exact_retry_after():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(3, 60.0, clock=clock)
    for _ in range(3):
        limiter.check("k")
    with pytest.raises(RateLimitExceeded) as info:
        limiter.check("k")
    assert info.value.limit == 3 and info.value.retry_after == 60

    clock.now += 59
    with pytest.raises(RateLimitExceeded) as info:
        limiter.check("k")
    assert info.value.retry_after == 1

    clock.now += 1  # the oldest event has just left the window
    limiter.check("k")
    assert limiter.rejected == 2


def test_limiter_isolates_identities():
    limiter = SlidingWindowLimiter(1, 60.0, clock=FakeClock())
    limiter.check("key:a")
    limiter.check("key:b")
    with pytest.raises(RateLimitExceeded):
        limiter.check("key:a")


def test_limiter_sweeps_drained_keys_when_table_is_full():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(1, 60.0, clock=clock)
    limiter.MAX_KEYS = 4
    for name in "abcd":
        limiter.check(name)
    clock.now += 61
    limiter.check("e")  # table full and every old key drained -> swept
    assert set(limiter._events) == {"e"}


@pytest.mark.parametrize("limit,window", [(0, 60.0), (-1, 60.0), (1, 0.0)])
def test_limiter_rejects_non_positive_parameters(limit, window):
    with pytest.raises(ValueError):
        SlidingWindowLimiter(limit, window)


# ---------------------------------------------------------- units: guard


def test_guard_rejects_overflow_without_blocking_and_releases_slots():
    guard = ConcurrencyGuard(2)
    with guard.slot(), guard.slot():
        assert guard.in_flight == 2
        with pytest.raises(ConcurrencyLimitExceeded) as info:
            with guard.slot():
                pytest.fail("a full guard must not admit a third holder")
        assert info.value.limit == 2
    assert guard.in_flight == 0 and guard.rejected == 1
    with guard.slot():  # fully released after the normal exit
        assert guard.in_flight == 1


def test_guard_releases_slot_when_the_body_raises():
    guard = ConcurrencyGuard(1)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            with guard.slot():
                raise RuntimeError("upstream exploded")
    assert guard.in_flight == 0 and guard.rejected == 0


def test_guard_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        ConcurrencyGuard(0)


# ----------------------------------------------------------- config surface


@pytest.mark.parametrize(
    "alias",
    [
        "MED_RATE_LIMIT_CONSULT_PER_MINUTE",
        "MED_RATE_LIMIT_IMAGE_PER_MINUTE",
        "MED_RATE_LIMIT_INGEST_PER_MINUTE",
        "MED_RATE_LIMIT_REBUILD_PER_HOUR",
        "MED_LLM_CONCURRENCY",
    ],
)
def test_config_rejects_non_positive_limits_at_startup(alias, monkeypatch):
    from pydantic import ValidationError

    from medcopilot.common.config import AppConfig

    monkeypatch.setenv(alias, "0")
    with pytest.raises(ValidationError):
        AppConfig(_env_file=None)

# ------------------------------------------------------------- HTTP: quotas


@pytest.fixture
def make_app(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import medcopilot.api.server as server_module
    from medcopilot.common.config import config

    def _make(*, raise_server_exceptions=True, **overrides):
        settings = {"rate_limit_enabled": True, "admin_api_key": None, "project_env": "development"}
        settings.update(overrides)
        for name, value in settings.items():
            monkeypatch.setattr(config, name, value)
        app = server_module.create_app()
        return app, TestClient(app, raise_server_exceptions=raise_server_exceptions)

    return _make


def _ask(client, query=PLAIN_QUERY, **kwargs):
    return client.post("/consultations", json={"query": query}, **kwargs)


def _ask_with_image(client):
    return client.post(
        "/consultations/with-image",
        data={"query": "这个药盒是什么药"},
        files={"image": ("box.jpg", JPEG, "image/jpeg")},
    )


def test_consultation_quota_returns_429_with_full_contract(make_app):
    _, client = make_app(rate_limit_consult_per_minute=3)
    for _ in range(3):
        assert _ask(client).status_code == 200
    blocked = _ask(client)
    assert blocked.status_code == 429
    detail = blocked.json()["detail"]
    assert detail["error"] == "rate_limited"
    assert detail["limit"] == 3 and detail["window_seconds"] == 60
    assert 1 <= detail["retry_after_seconds"] <= 60
    assert "120" in detail["emergency_notice"]
    assert blocked.headers["Retry-After"] == str(detail["retry_after_seconds"])
    assert blocked.headers["X-RateLimit-Limit"] == "3"
    assert blocked.headers["X-RateLimit-Remaining"] == "0"


def test_emergency_triage_is_never_throttled(make_app):
    _, client = make_app(rate_limit_consult_per_minute=1)
    assert _ask(client).status_code == 200
    assert _ask(client).status_code == 429
    for _ in range(5):
        response = _ask(client, EMERGENCY_QUERY)
        assert response.status_code == 200 and response.json()["is_emergency"] is True
    assert client.get("/health").json()["rate_limited_total"] == 1  # only the ordinary rejection


def test_auth_failures_do_not_consume_the_callers_quota(make_app):
    _, client = make_app(rate_limit_consult_per_minute=2, admin_api_key=ADMIN_KEY)
    wrong = {"Authorization": "Bearer wrong"}
    assert [_ask(client, headers=wrong).status_code for _ in range(10)] == [401] * 10
    assert _ask(client, headers=AUTH).status_code == 200
    assert _ask(client, headers=AUTH).status_code == 200
    assert _ask(client, headers=AUTH).status_code == 429


def test_oversized_body_413_does_not_consume_quota(make_app, monkeypatch):
    import medcopilot.api.server as server_module

    monkeypatch.setattr(server_module, "MAX_BODY_BYTES", 256)
    _, client = make_app(rate_limit_consult_per_minute=2)
    assert [_ask(client, "头痛" * 200).status_code for _ in range(4)] == [413] * 4
    assert _ask(client).status_code == 200
    assert _ask(client).status_code == 200
    assert _ask(client).status_code == 429


def test_quotas_are_per_endpoint(make_app, monkeypatch):
    from medcopilot.agents.ocr_tool import OCRTool

    class OkEngine:
        def ocr(self, path, cls=True):
            return [[{"rec_text": "阿莫西林"}]]

    monkeypatch.setattr(OCRTool, "_load_engine", lambda self: OkEngine())
    _, client = make_app(rate_limit_consult_per_minute=1, rate_limit_image_per_minute=1)
    assert _ask(client).status_code == 200
    assert _ask(client).status_code == 429
    assert _ask_with_image(client).status_code == 200  # own counter, untouched by /consultations
    assert _ask_with_image(client).status_code == 429


def test_ingest_quota(make_app):
    _, client = make_app(rate_limit_ingest_per_minute=2, admin_api_key=ADMIN_KEY)

    def ingest():
        form = {"raw_text": LABEL, "drug_name": "阿莫西林胶囊"}
        return client.post("/documents/drug-label", data=form, headers=AUTH)

    assert ingest().status_code == 200
    assert ingest().status_code == 200
    blocked = ingest()
    assert blocked.status_code == 429
    assert blocked.json()["detail"]["limit"] == 2


def test_rebuild_quota_uses_an_hourly_window(make_app):
    _, client = make_app(rate_limit_rebuild_per_hour=1, admin_api_key=ADMIN_KEY)
    # No persistent corpus in this app, so the rebuild itself is refused, but
    # the attempt still spends the hourly token.
    assert client.post("/index/rebuild", headers=AUTH).status_code != 429
    blocked = client.post("/index/rebuild", headers=AUTH)
    assert blocked.status_code == 429
    detail = blocked.json()["detail"]
    assert detail["window_seconds"] == 3600 and detail["retry_after_seconds"] > 60


def test_disabled_limiter_never_returns_429(make_app):
    _, client = make_app(rate_limit_enabled=False, rate_limit_consult_per_minute=1)
    assert [_ask(client).status_code for _ in range(5)] == [200] * 5
    health = client.get("/health").json()
    assert health["rate_limit_enabled"] is False and health["rate_limited_total"] == 0

# ------------------------------------------------------- HTTP: concurrency


class _Blocker:
    """Stands in for ``workflow.run``: parks every caller until released."""

    def __init__(self, real_run):
        self.real_run = real_run
        self.entered = threading.Semaphore(0)
        self.release = threading.Event()

    def __call__(self, *args, **kwargs):
        self.entered.release()
        assert self.release.wait(timeout=30), "test forgot to release the parked LLM calls"
        return self.real_run(*args, **kwargs)


def _wait_in_flight(client, expected, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/health").json()["llm_in_flight"] == expected:
            return
        time.sleep(0.02)
    pytest.fail(f"llm_in_flight never reached {expected}")


def _run_threads(targets):
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    return threads


def test_saturated_guard_fails_fast_while_emergency_and_health_stay_live(make_app, monkeypatch):
    app, client = make_app(llm_concurrency=2, rate_limit_enabled=False)
    workflow = app.state.runtime_holder.runtime.workflow
    blocker = _Blocker(workflow.run)
    monkeypatch.setattr(workflow, "run", blocker)

    results: list[int] = []
    holders = _run_threads([lambda: results.append(_ask(client).status_code)] * 2)
    try:
        for _ in range(2):
            assert blocker.entered.acquire(timeout=10)

        overflow = _ask(client)
        assert overflow.status_code == 503
        detail = overflow.json()["detail"]
        assert detail["error"] == "server_busy" and detail["limit"] == 2
        assert "120" in detail["emergency_notice"]
        assert overflow.headers["Retry-After"] == "1"
        assert _ask_with_image(client).status_code == 503  # same guard covers the OCR path

        emergency = _ask(client, EMERGENCY_QUERY)
        assert emergency.status_code == 200 and emergency.json()["is_emergency"] is True
        health = client.get("/health").json()
        assert health["llm_in_flight"] == 2 and health["overloaded_total"] == 2
    finally:
        blocker.release.set()
        for holder in holders:
            holder.join(timeout=30)
    assert results == [200, 200]
    assert _ask(client).status_code == 200  # slots were handed back


def test_same_session_waiters_count_against_the_guard(make_app, monkeypatch):
    app, client = make_app(llm_concurrency=2, rate_limit_enabled=False)
    session_id = _ask(client).json()["session_id"]
    workflow = app.state.runtime_holder.runtime.workflow
    blocker = _Blocker(workflow.run)
    monkeypatch.setattr(workflow, "run", blocker)

    results: list[int] = []

    def follow_up():
        response = client.post("/consultations", json={"query": PLAIN_QUERY, "session_id": session_id})
        results.append(response.status_code)

    holders = _run_threads([follow_up, follow_up])
    try:
        assert blocker.entered.acquire(timeout=10)  # first holds the session lock inside run()
        _wait_in_flight(client, 2)  # second is parked on that lock but still owns a slot
        assert _ask(client).status_code == 503
    finally:
        blocker.release.set()
        for holder in holders:
            holder.join(timeout=30)
    assert results == [200, 200]


def test_slot_is_released_when_the_workflow_raises(make_app, monkeypatch):
    app, client = make_app(llm_concurrency=1, rate_limit_enabled=False, raise_server_exceptions=False)
    workflow = app.state.runtime_holder.runtime.workflow

    def explode(*args, **kwargs):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(workflow, "run", explode)
    # 500 three times, never 503: each failed request gave its slot back.
    assert [_ask(client).status_code for _ in range(3)] == [500] * 3
    assert client.get("/health").json()["llm_in_flight"] == 0
