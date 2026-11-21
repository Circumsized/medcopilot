"""W2.1/W2.2/W2.4 contracts for explicit LLM mode and bounded retries."""
from __future__ import annotations

import json
from collections import Counter

import httpx
import pytest

from medcopilot.common.config import config
from medcopilot.llm.unified_llm import UnifiedLLM


@pytest.fixture(autouse=True)
def _clean_llm_environment(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setattr(config, "mock_llm", False)
    monkeypatch.setattr(config, "project_env", "development")
    if hasattr(config, "allow_mock_in_prod"):
        monkeypatch.setattr(config, "allow_mock_in_prod", False)


def _mock_reason(llm: UnifiedLLM) -> str | None:
    return llm._mock_reason()


def test_mock_reason_distinguishes_provider_env_placeholder_and_missing_key(monkeypatch):
    assert _mock_reason(UnifiedLLM(provider="mock")) == "provider_mock"

    monkeypatch.setenv("MOCK_LLM", "yes")
    assert _mock_reason(UnifiedLLM(provider="openai_compatible", api_key="sk-live")) == "mock_env"
    monkeypatch.delenv("MOCK_LLM")

    assert _mock_reason(UnifiedLLM(api_key="your-api-key-here")) == "placeholder_key"
    assert _mock_reason(UnifiedLLM(api_key="")) == "missing_key"
    assert _mock_reason(UnifiedLLM(api_key="", api_base="http://localhost:11434/v1")) is None


def test_provider_specific_credentials_determine_mock_reason(monkeypatch):
    assert _mock_reason(UnifiedLLM(provider="qianfan")) == "missing_key"
    monkeypatch.setattr(config, "qianfan_ak", "ak-live")
    monkeypatch.setattr(config, "qianfan_sk", "sk-live")
    assert _mock_reason(UnifiedLLM(provider="qianfan")) is None

    monkeypatch.setattr(config, "zhipuai_api_key", "your_zhipuai_api_key")
    assert _mock_reason(UnifiedLLM(provider="zhipu")) == "placeholder_key"
    monkeypatch.setattr(config, "zhipuai_api_key", "zhipu-live")
    assert _mock_reason(UnifiedLLM(provider="zhipu")) is None


def test_production_unconfigured_llm_returns_error_not_mock_text(monkeypatch):
    monkeypatch.setattr(config, "project_env", "production")
    invocation = UnifiedLLM(api_key="").invoke_with_status("患者提问")
    assert invocation.mode == "error"
    assert invocation.reason == "missing_key"
    assert invocation.error_name == "UnconfiguredInProduction"
    assert invocation.mocked is False
    assert "模拟诊疗意见" not in invocation.text


def test_production_mock_can_only_be_enabled_by_explicit_config(monkeypatch):
    monkeypatch.setattr(config, "project_env", "prod")
    monkeypatch.setattr(config, "allow_mock_in_prod", True)
    invocation = UnifiedLLM(provider="mock").invoke_with_status("问题")
    assert invocation.mode == "mock"
    assert invocation.reason == "provider_mock"
    assert invocation.mocked is True


def test_retry_config_defaults_are_shared_with_app_config():
    from medcopilot.common.config import AppConfig

    values = {name: field.default for name, field in AppConfig.model_fields.items()}
    llm = UnifiedLLM()
    assert llm.max_tokens == values["llm_max_tokens"] == 2048
    assert llm.timeout == values["llm_timeout"] == 60.0
    assert llm.max_attempts == values["llm_max_attempts"] == 3
    assert llm.backoff_base == values["llm_backoff_base"] == 1.0
    assert llm.backoff_cap == values["llm_backoff_cap"] == 20.0
    assert llm.retry_deadline == values["llm_retry_deadline"] == 45.0


def test_legacy_max_retries_and_retry_backoff_keep_total_attempt_semantics():
    llm = UnifiedLLM(max_retries=2, retry_backoff=0)
    assert llm._effective_max_attempts() == 2
    assert llm._effective_backoff_base() == 0


@pytest.mark.parametrize(
    "status,retryable,diagnostic",
    [
        (400, False, "check_request_parameters"),
        (401, False, "check_LLM_API_KEY"),
        (403, False, "check_LLM_API_KEY"),
        (404, False, "check_request_parameters"),
        (408, True, "http_408"),
        (409, True, "http_409"),
        (422, False, "check_request_parameters"),
        (429, True, "http_429"),
        (500, True, "http_500"),
        (599, True, "http_599"),
    ],
)
def test_http_error_classification(status, retryable, diagnostic):
    request = httpx.Request("POST", "https://llm.example.test/chat")
    response = httpx.Response(status, request=request)
    error = httpx.HTTPStatusError("provider response", request=request, response=response)
    assert UnifiedLLM._classify_llm_error(error) == (retryable, diagnostic)


def test_network_and_truncated_json_are_retryable_but_runtime_errors_are_not():
    request = httpx.Request("POST", "https://llm.example.test/chat")
    assert UnifiedLLM._classify_llm_error(httpx.ConnectError("offline", request=request))[0] is True
    assert UnifiedLLM._classify_llm_error(json.JSONDecodeError("truncated", "{", 1))[0] is True
    assert UnifiedLLM._classify_llm_error(RuntimeError("missing provider credentials"))[0] is False


def _mock_http_client(monkeypatch, outcomes, calls: Counter, *, retry_after=None):
    original_client = httpx.Client
    remaining = list(outcomes)

    def handler(request):
        calls["count"] += 1
        result = remaining.pop(0)
        if isinstance(result, Exception):
            raise result
        status, body = result
        headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
        return httpx.Response(status, json=body, headers=headers, request=request)

    transport = httpx.MockTransport(handler)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)
    monkeypatch.setattr(UnifiedLLM, "_should_mock", lambda self: False)
    monkeypatch.setattr(UnifiedLLM, "_bundle", lambda self, prompt, stream: {
        "url": "https://llm.example.test/chat",
        "headers": {},
        "payload": {},
        "params": None,
    })


def test_401_fails_once_with_actionable_diagnostic_and_no_sleep(monkeypatch):
    calls = Counter()
    sleeps = []
    _mock_http_client(monkeypatch, [(401, {"error": "unauthorized"})] * 3, calls)
    monkeypatch.setattr("medcopilot.llm.unified_llm.time.sleep", sleeps.append)
    llm = UnifiedLLM(max_attempts=3)
    result = llm.invoke_with_status("问题")
    assert calls["count"] == 1
    assert sleeps == []
    assert result.error_name == "HTTPStatusError"
    assert result.diagnostic == "check_LLM_API_KEY"


def test_500_retries_up_to_max_attempts_with_full_jitter_bounds(monkeypatch):
    calls, sleeps, bounds = Counter(), [], []
    _mock_http_client(
        monkeypatch,
        [(500, {"error": "temporary"}), (500, {"error": "temporary"}),
         (200, {"choices": [{"message": {"content": "ready"}}]})],
        calls,
    )
    monkeypatch.setattr("medcopilot.llm.unified_llm.time.sleep", sleeps.append)

    def choose_low(low, high):
        bounds.append((low, high))
        return low

    monkeypatch.setattr("medcopilot.llm.unified_llm.random.uniform", choose_low)
    result = UnifiedLLM(max_attempts=3, backoff_base=1, backoff_cap=20).invoke_with_status("问题")
    assert result.text == "ready" and result.mode == "live"
    assert calls["count"] == 3
    assert [high for _, high in bounds] == [1, 2]
    assert sleeps == [0, 0]


def test_429_retry_after_is_capped_and_jittered(monkeypatch):
    calls, sleeps, bounds = Counter(), [], []
    _mock_http_client(
        monkeypatch,
        [(429, {"error": "rate limited"}),
         (200, {"choices": [{"message": {"content": "ready"}}]})],
        calls,
        retry_after=7,
    )
    monkeypatch.setattr("medcopilot.llm.unified_llm.time.sleep", sleeps.append)
    monkeypatch.setattr(
        "medcopilot.llm.unified_llm.random.uniform",
        lambda low, high: bounds.append((low, high)) or high,
    )
    result = UnifiedLLM(max_attempts=2, backoff_cap=20).invoke_with_status("问题")
    assert result.text == "ready" and calls["count"] == 2
    assert bounds == [(0.0, 7.0)] and sleeps == [7.0]


def test_retry_deadline_stops_before_another_provider_attempt(monkeypatch):
    calls, sleeps = Counter(), []
    _mock_http_client(monkeypatch, [(500, {"error": "temporary"})] * 3, calls)
    monkeypatch.setattr("medcopilot.llm.unified_llm.time.sleep", sleeps.append)
    monkeypatch.setattr("medcopilot.llm.unified_llm.random.uniform", lambda low, high: high)
    monkeypatch.setattr("medcopilot.llm.unified_llm.time.monotonic", lambda: 100.0)
    llm = UnifiedLLM(max_attempts=3, backoff_base=2, retry_deadline=0.5)
    result = llm.invoke_with_status("问题")
    assert calls["count"] == 1
    assert result.error_name == "RetryBudgetExceeded"
    assert result.diagnostic == "retry_deadline_exceeded"


def test_empty_provider_response_retries_within_budget_then_reports_error(monkeypatch):
    """空响应(200 无内容)按可重试故障处理:消耗有界预算而非一次抖动即终局;
    持续为空时是真实错误状态(mode=error/EmptyLLMResponse),绝不伪称成功。"""
    calls = Counter()
    _mock_http_client(monkeypatch, [(200, {"choices": []})] * 3, calls)
    result = UnifiedLLM(max_attempts=3, backoff_base=0).invoke_with_status("问题")
    assert calls["count"] == 3
    assert result.mode == "error"
    assert result.error_name == "EmptyLLMResponse"
    assert result.diagnostic == "empty_provider_response"
    assert result.mocked is False


def test_empty_provider_response_recovers_on_retry(monkeypatch):
    calls = Counter()
    _mock_http_client(
        monkeypatch,
        [(200, {"choices": []}), (200, {"choices": [{"message": {"content": "ready"}}]})],
        calls,
    )
    result = UnifiedLLM(max_attempts=3, backoff_base=0).invoke_with_status("问题")
    assert result.text == "ready" and result.mode == "live"
    assert calls["count"] == 2


def test_api_health_reports_mock_reason_and_effective_model(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    monkeypatch.setattr(config, "llm_api_key", "")
    monkeypatch.setattr(config, "llm_model", "test-model")
    client = TestClient(create_app())
    health = client.get("/health").json()
    assert health["llm_mode"] == "mock"
    assert health["llm_mode_reason"] == "missing_key"
    assert health["llm_model_effective"] == "mock:test-model"
    assert health["llm_mock_fallback_total"] == 0


def test_api_health_reports_unconfigured_production_as_error(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    monkeypatch.setattr(config, "project_env", "production")
    monkeypatch.setattr(config, "admin_api_key", "secret")
    monkeypatch.setattr(config, "llm_api_key", "")
    health = TestClient(create_app()).get("/health", headers={"Authorization": "Bearer secret"}).json()
    assert health["llm_mode"] == "error"
    assert health["llm_mode_reason"] == "missing_key"
    assert health["llm_model_effective"] == "mock-blocked"


def test_health_omits_llm_inventory_for_anonymous_production_caller(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from medcopilot.api.server import create_app

    monkeypatch.setattr(config, "project_env", "production")
    monkeypatch.setattr(config, "admin_api_key", "secret")
    health = TestClient(create_app()).get("/health").json()
    assert "llm_mode" not in health
    assert "llm_model_effective" not in health


def test_qianfan_concurrent_first_token_refresh_is_single_flight(monkeypatch):
    import threading
    import time as real_time

    monkeypatch.setattr(config, "qianfan_ak", "ak-live")
    monkeypatch.setattr(config, "qianfan_sk", "sk-live")
    fetches = Counter()

    def handler(request):
        fetches["count"] += 1
        real_time.sleep(0.05)  # widen the race window so unsynchronized code double-fetches
        return httpx.Response(200, json={"access_token": "tok", "expires_in": 1800}, request=request)

    original_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **k: original_client(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )
    llm = UnifiedLLM(provider="qianfan")
    barrier = threading.Barrier(8)
    tokens = []

    def worker():
        barrier.wait()
        tokens.append(llm._qianfan_token())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert tokens == ["tok"] * 8
    assert fetches["count"] == 1
