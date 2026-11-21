"""W6.3 prompt 角色分离契约:str 兼容 + list role 白名单 + provider 透传。

同时固定本轮顺手修复的 UnifiedLLM 硬伤(同一职责只留一份实现):
- 模式分类(实例 `_mock_reason` 与 `/health` 的 `classify_mode`)共用同一 credential 判定;
- 占位凭据判定只走 `_is_placeholder_key` 一个入口;
- 未知 provider fail-closed(绝不静默按 openai_compatible 发包);
- 重试的 sleep 不超过剩余时间预算(429 Retry-After 仍受 backoff_cap 封顶);
- 每次调用的真值只在返回值(LLMInvocation)上,实例不再写 per-call 可变状态。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import httpx
import pytest

from medcopilot.common.config import config
from medcopilot.llm.unified_llm import UnifiedLLM
from medcopilot.workflow.prompt_budget import EVIDENCE_HEADER, QUERY_HEADER


@pytest.fixture(autouse=True)
def _clean_llm_environment(monkeypatch):
    monkeypatch.delenv("MOCK_LLM", raising=False)
    monkeypatch.setattr(config, "mock_llm", False)
    monkeypatch.setattr(config, "project_env", "development")
    if hasattr(config, "allow_mock_in_prod"):
        monkeypatch.setattr(config, "allow_mock_in_prod", False)


def _bundle_messages(llm: UnifiedLLM, messages: Any) -> list[dict[str, str]]:
    return llm._bundle(messages, stream=False)["payload"]["messages"]


# ------------------------------- 角色白名单 ---------------------------------


def test_string_prompt_keeps_legacy_two_message_shape():
    messages = UnifiedLLM()._normalize_messages("患者提问")
    # 旧 str 路径逐字节保持:仍是 [通用 system, user=原 prompt]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "患者提问"
    assert "临床医生助手" in messages[0]["content"]


def test_message_list_round_trips_roles_content_and_order():
    messages = [
        {"role": "system", "content": "原则"},
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
    ]
    normalized = UnifiedLLM()._normalize_messages(messages)
    assert normalized == messages
    # 返回新 dict:调用方事后改自己的 list 不会影响已归一化的请求
    assert all(out is not src for out, src in zip(normalized, messages))


@pytest.mark.parametrize("bad", [
    [{"role": "developer", "content": "x"}],   # 未知角色
    [{"role": "system"}],                      # 缺 content
    [{"role": "system", "content": "   "}],    # 空 content
    ["just a string"],                         # 非 mapping
    [],                                        # 空列表
])
def test_invalid_message_lists_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        UnifiedLLM()._normalize_messages(bad)


def test_role_whitespace_is_normalized_before_validation():
    normalized = UnifiedLLM()._normalize_messages([{"role": " user ", "content": "问题"}])
    assert normalized == [{"role": "user", "content": "问题"}]


def test_all_providers_carry_the_caller_supplied_roles(monkeypatch):
    monkeypatch.setattr(config, "qianfan_ak", "ak-live")
    monkeypatch.setattr(config, "qianfan_sk", "sk-live")
    monkeypatch.setattr(UnifiedLLM, "_qianfan_token", lambda self: "access-token")
    messages = [
        {"role": "system", "content": "system=原则部分"},
        {"role": "user", "content": "user=不可信历史/OCR/证据/query"},
    ]
    for provider in ("openai_compatible", "qianfan", "zhipu"):
        llm = UnifiedLLM(provider=provider, api_key="live-key")
        assert _bundle_messages(llm, messages) == messages, provider


def test_unknown_provider_fails_closed_before_any_request(monkeypatch):
    def boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("network attempted with an unknown provider")

    monkeypatch.setattr(httpx, "Client", boom)
    llm = UnifiedLLM(provider="nonsense", api_key="live-key")
    assert llm._mock_reason() == "unknown_provider"
    assert UnifiedLLM.classify_mode("nonsense", "live-key", "https://api.example.com/v1", False) == (
        "error", "unknown_provider",
    )
    with pytest.raises(RuntimeError):
        llm._bundle([{"role": "user", "content": "hi"}], stream=False)
    # dev 下降级到 mock,生产 fail-closed:两条路都不发包
    assert llm.invoke_with_status("问题").mocked is True


def test_mock_reply_extracts_query_from_a_message_list():
    from medcopilot.workflow.agent_graph import UNTRUSTED_END, UNTRUSTED_START

    llm = UnifiedLLM(provider="mock")
    reply = llm._mock_reply([
        {"role": "system", "content": "你是一名资深的临床医生助手。原则：仅依据资料作答。"},
        {"role": "user", "content": (
            f"{EVIDENCE_HEADER}\n{UNTRUSTED_START}\n【用法用量】每日三次。{UNTRUSTED_END}\n\n"
            f"{QUERY_HEADER}\n{UNTRUSTED_START}\n头痛可以吃布洛芬吗{UNTRUSTED_END}"
        )},
    ])
    assert "头痛可以吃布洛芬吗" in reply
    assert "你是一名" not in reply


def test_stream_accepts_messages_and_mock_mode_is_dev_only(monkeypatch):
    monkeypatch.setattr(config, "project_env", "development")
    llm = UnifiedLLM(provider="mock")
    chunks = list(llm.stream([{"role": "user", "content": "问题"}]))
    assert "".join(chunks) == llm.invoke([{"role": "user", "content": "问题"}])
    monkeypatch.setattr(config, "project_env", "production")
    with pytest.raises(RuntimeError):
        list(llm.stream("问题"))


# --------------------------------- kwargs ----------------------------------


def test_unknown_kwargs_raise_type_error():
    llm = UnifiedLLM(provider="mock")
    with pytest.raises(TypeError):
        llm.invoke("问题", temperature=0.2)
    with pytest.raises(TypeError):
        llm.invoke_with_status("问题", bogus=1)
    with pytest.raises(TypeError):
        list(llm.stream("问题", bogus=1))


def test_stop_stays_a_supported_kwarg():
    llm = UnifiedLLM(provider="mock")
    assert llm.invoke("问题", stop=["\n"]) == llm.invoke_with_status("问题", stop=["\n"]).text


# --------------------------- 模式分类去重 / fail-closed ---------------------


def test_placeholder_detection_goes_through_the_single_shared_predicate(monkeypatch):
    """任一 provider 的占位凭据判定都必须复用 `_is_placeholder_key`,
    而不是各处内联一份子串判断(改判定标准时只改一处)。"""
    monkeypatch.setattr(config, "qianfan_ak", "ak-live")
    monkeypatch.setattr(config, "qianfan_sk", "sk-live")
    monkeypatch.setattr(config, "zhipuai_api_key", "zhipu-live")
    monkeypatch.setattr(UnifiedLLM, "_is_placeholder_key", classmethod(lambda cls, value: True))
    assert UnifiedLLM(api_key="sk-live")._mock_reason() == "placeholder_key"
    assert UnifiedLLM(provider="qianfan")._mock_reason() == "placeholder_key"
    assert UnifiedLLM(provider="zhipu")._mock_reason() == "placeholder_key"


def test_classify_mode_agrees_with_the_instance_decision_for_every_provider(monkeypatch):
    monkeypatch.setattr(config, "qianfan_ak", "ak-live")
    monkeypatch.setattr(config, "qianfan_sk", "sk-live")
    monkeypatch.setattr(config, "zhipuai_api_key", "zhipu-live")
    for provider in UnifiedLLM.SUPPORTED_PROVIDERS:
        llm = UnifiedLLM(provider=provider, api_key="live-key")
        assert UnifiedLLM.classify_mode(provider, "live-key", "https://api.example.com/v1", False) == ("live", None)
        assert llm._mock_reason() is None

    # dev + 无凭据 ⇒ mock(可观测降级);production + 无凭据 ⇒ error(不冒充已联通)
    for project_env, expected in (("development", ("mock", "missing_key")), ("production", ("error", "missing_key"))):
        monkeypatch.setattr(config, "project_env", project_env)
        assert UnifiedLLM.classify_mode("openai_compatible", "", "https://api.example.com/v1", False) == expected


def test_local_openai_endpoint_is_live_without_credentials():
    llm = UnifiedLLM(api_key="", api_base="http://localhost:11434/v1")
    assert llm._mock_reason() is None
    assert UnifiedLLM.classify_mode("openai_compatible", "", "http://localhost:11434/v1", False) == ("live", None)


# ------------------------------ 重试时间预算 --------------------------------


def _install_fake_client(monkeypatch, outcomes, *, retry_after=None):
    calls: Counter = Counter()
    remaining = list(outcomes)

    def handler(request):
        calls["count"] += 1
        result = remaining.pop(0) if remaining else (200, {"choices": [{"message": {"content": "ready"}}]})
        if isinstance(result, Exception):
            raise result
        status, body = result
        headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
        return httpx.Response(status, json=body, headers=headers, request=request)

    original = httpx.Client
    monkeypatch.setattr(
        httpx, "Client",
        lambda *a, **k: original(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )
    monkeypatch.setattr(UnifiedLLM, "_should_mock", lambda self: False)
    monkeypatch.setattr(
        UnifiedLLM, "_bundle",
        lambda self, messages, stream: {"url": "https://llm.example.test/chat", "headers": {}, "payload": {}, "params": None},
    )
    return calls


def test_retry_after_never_sleeps_past_the_time_budget(monkeypatch):
    """429 的 Retry-After 受 backoff_cap 封顶,且 sleep 不得超过剩余时间预算——
    否则一次"稍后重试" header 就能把整个请求预算烧光。"""
    calls = _install_fake_client(monkeypatch, [(429, {"error": "rate limited"})] * 3, retry_after=30)
    sleeps: list[float] = []
    monkeypatch.setattr("medcopilot.llm.unified_llm.time.sleep", sleeps.append)
    monkeypatch.setattr("medcopilot.llm.unified_llm.random.uniform", lambda low, high: high)
    monkeypatch.setattr("medcopilot.llm.unified_llm.time.monotonic", lambda: 100.0)
    result = UnifiedLLM(max_attempts=3, backoff_cap=20, retry_deadline=0.5).invoke_with_status("问题")
    # 天花板已被时间预算压到 <= remaining:不再发起下一次请求,也不睡觉
    assert calls["count"] == 1
    assert sleeps == []
    assert result.error_name == "RetryBudgetExceeded"
    assert result.diagnostic == "retry_deadline_exceeded"


def test_retry_deadline_is_measured_across_attempts(monkeypatch):
    """时间预算按墙钟累计:单次 attempt 很快,累计 sleep 到顶即停,不再发包。"""
    calls = _install_fake_client(monkeypatch, [(500, {"error": "temporary"})] * 8)
    clock = {"now": 100.0}
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        clock["now"] += seconds
        sleeps.append(seconds)

    monkeypatch.setattr("medcopilot.llm.unified_llm.time.sleep", fake_sleep)
    monkeypatch.setattr("medcopilot.llm.unified_llm.time.monotonic", lambda: clock["now"])
    monkeypatch.setattr("medcopilot.llm.unified_llm.random.uniform", lambda low, high: high)
    result = UnifiedLLM(max_attempts=8, backoff_base=1, backoff_cap=20, retry_deadline=2.5).invoke_with_status("问题")
    assert result.error_name == "RetryBudgetExceeded"
    assert sum(sleeps) <= 2.5  # 预算内睡觉,绝不超总预算
    assert calls["count"] == 2  # 到点即停,不会把 8 次 attempts 耗尽


# --------------------------- per-call 真值只在返回值 ------------------------


def test_mode_probes_do_not_write_per_call_instance_state():
    llm = UnifiedLLM(provider="mock")
    before = dict(llm.__pydantic_private__)
    llm._mock_reason()
    llm._should_mock()
    assert dict(llm.__pydantic_private__) == before


def test_mock_and_error_outcomes_ride_the_return_value_only():
    """mock 降级与错误都不依赖实例上遗留的"上一次状态"作为本次判据。"""
    llm = UnifiedLLM(provider="mock")
    first = llm.invoke_with_status("问题")
    assert first.mocked is True and first.reason == "provider_mock"
    second = llm.invoke_with_status("问题")
    assert second.reason == "provider_mock"
    assert llm.mock_fallback_total == 2  # 聚合计数器仍是可观测诊断信号
