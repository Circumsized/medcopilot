from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field, PrivateAttr

from medcopilot.common.config import config
from medcopilot.common.logger import logger

# W6.3: 调用方可传 str(旧路径)或 role 已校验的 message list(新路径)。
LLMMessage = Mapping[str, str]
PromptInput = str | Sequence[LLMMessage]

# Module-level: pydantic v2 would treat an underscore-prefixed *class* attribute
# as a private-attr declaration, shadowing the compiled pattern.
# Covers query-params, Bearer (including +/= Base64 chars), Basic auth, and JSON key:value.
# ``token`` (bare) is included so a provider error surfacing "token":"..." or ?token=...
# does not leak the secret (CWE-209); ``access_token`` still matches first in the alternation.
_SECRET_KEY_NAME = (
    r"(?:[A-Za-z0-9_.-]*[_-])?"
    r"(?:access[_-]?token|client[_-]?secret|client[_-]?id|api[_-]?key|apikey|signing[_-]?key|qianfan[_-]?(?:sk|ak)|secret|token|key|password)"
)
_SECRET_PATTERN = re.compile(
    rf"\b({_SECRET_KEY_NAME})=([^&\s\"']+)"
    rf"|[\"']({_SECRET_KEY_NAME})[\"']\s*:\s*[\"']([^\"']+)[\"']"
    r"|\bBearer\s+[A-Za-z0-9._\-+=/]+"
    r"|\bBasic\s+[A-Za-z0-9._\-+=/]+",
    re.IGNORECASE,
)


class _HttpxSecretRedactionFilter(logging.Filter):
    """Redact credentials in httpx log records (CWE-532, second layer).

    httpx logs the full request URL at INFO/DEBUG; for query-token providers
    (Qianfan ``?access_token=...``) that string carries the live credential.
    ``UnifiedLLM._redact`` already scrubs *responses*; this filter scrubs the
    *transport's own logging* so ``LOG_LEVEL=DEBUG`` cannot leak the secret
    either. Applied to the rendered message so both msg+args forms are covered.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed args
            return True
        redacted = _SECRET_PATTERN.sub(
            lambda match: (
                f"{match.group(1)}=***"
                if match.group(1)
                else f'"{match.group(3)}":"***"'
                if match.group(3)
                else ("Bearer ***" if match.group(0).lower().startswith("bearer") else "Basic ***")
            ),
            message,
        )
        if redacted != message:
            record.msg = redacted
            record.args = None
        return True


def install_httpx_secret_filter() -> logging.Filter:
    """Attach the redaction filter to the httpx logger (idempotent).

    The filter object is returned so tests can exercise it directly.
    """
    httpx_logger = logging.getLogger("httpx")
    for existing in httpx_logger.filters:
        if isinstance(existing, _HttpxSecretRedactionFilter):
            return existing
    redactor = _HttpxSecretRedactionFilter()
    httpx_logger.addFilter(redactor)
    return redactor


install_httpx_secret_filter()


@dataclass(frozen=True)
class LLMInvocation:
    """One ``invoke`` outcome: text plus immutable per-call diagnostics."""

    text: str
    error_name: str | None = None
    mocked: bool = False
    # live | mock | error; reason is a stable, non-sensitive reason code.
    mode: str = "live"
    reason: str | None = None
    diagnostic: str | None = None


class _EmptyProviderResponseError(Exception):
    """200 OK with no extractable content: transient, so it flows through the
    bounded retry machinery instead of ending the invocation on the first
    glitch. An exhausted budget still surfaces a real EmptyLLMResponse error.
    """


class UnifiedLLM(BaseModel):
    """Unified clinical LLM client.

    Providers
    ---------
    * ``openai_compatible``  — OpenAI / Qwen / DeepSeek / vLLM / Ollama (default)
    * ``qianfan``            — Baidu ERNIE via access-token endpoint
    * ``zhipu``              — ZhipuAI GLM via API-key bearer header

    W6.3 prompt role separation: ``invoke``/``stream`` accept either the legacy
    single string (wrapped verbatim into ``[system preamble, user prompt]``) or a
    role-validated message list, which every provider payload carries through
    unchanged. ``str`` compatibility is byte-for-byte preserved.

    In development, missing/placeholder credentials and explicit mock settings
    use a deterministic mock with an observable reason code. Production fails
    closed on those cases unless ``ALLOW_MOCK_IN_PRODUCTION=1`` is explicitly
    set. Retryable provider errors use bounded exponential full jitter that never
    sleeps past the invocation's time budget.
    """

    # W6.3: role 白名单(严格:归一化后必须精确命中,未知角色绝不静默降级)。
    MESSAGE_ROLES: ClassVar[frozenset[str]] = frozenset({"system", "user", "assistant"})
    # 已知 provider;未知值 fail-closed(不静默按 openai_compatible 发包)。
    SUPPORTED_PROVIDERS: ClassVar[frozenset[str]] = frozenset({"openai_compatible", "qianfan", "zhipu"})
    # invoke 接受的显式关键字参数;其余一律 TypeError(调用方拼错参数不再被静默吞掉)。
    KNOWN_INVOKE_KWARGS: ClassVar[frozenset[str]] = frozenset({"stop"})
    # 旧 str 路径的 system 前言(逐字节保持历史行为)。
    SYSTEM_PREAMBLE: ClassVar[str] = "你是一名严谨、专业的临床医生助手。所有意见仅供医学参考。"

    provider: str = Field(default_factory=lambda: config.llm_provider)
    api_key: str = Field(default_factory=lambda: config.llm_api_key)
    api_base: str = Field(default_factory=lambda: config.llm_api_base)
    model_name: str = Field(default_factory=lambda: config.llm_model)
    temperature: float = Field(default_factory=lambda: config.llm_temperature)
    max_tokens: int = Field(default_factory=lambda: config.llm_max_tokens, gt=0)
    timeout: float = Field(default_factory=lambda: config.llm_timeout, gt=0)
    max_attempts: int = Field(default_factory=lambda: config.llm_max_attempts, gt=0)
    backoff_base: float = Field(default_factory=lambda: config.llm_backoff_base, ge=0)
    backoff_cap: float = Field(default_factory=lambda: config.llm_backoff_cap, gt=0)
    retry_deadline: float = Field(default_factory=lambda: config.llm_retry_deadline, gt=0)
    enforce_mode: bool = Field(default=False, exclude=True)
    # Constructor compatibility: historically max_retries was the total number
    # of calls, despite its name. Explicit legacy values override max_attempts.
    max_retries: int | None = Field(default=None, ge=1, exclude=True)
    retry_backoff: float | None = Field(default=None, ge=0, exclude=True)
    _token_cache: tuple[str, float] | None = PrivateAttr(default=None)
    _token_refresh_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    # 聚合诊断(不是 per-call 真值):/health 的 llm_mock_fallback_total 读它。
    _mock_fallback_total: int = PrivateAttr(default=0)
    _state_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def mock_fallback_total(self) -> int:
        with self._state_lock:
            return self._mock_fallback_total

    def _record_mock_fallback(self) -> None:
        with self._state_lock:
            self._mock_fallback_total += 1

    @staticmethod
    def _flag_enabled(value: str | None) -> bool:
        return (value or "").strip().lower() in {"1", "true", "yes", "on"}

    def _mock_enabled(self) -> bool:
        # Honor both MOCK_LLM env var and AppConfig (which also reads .env).
        return self.provider == "mock" or self._flag_enabled(os.getenv("MOCK_LLM")) or bool(config.mock_llm)

    @classmethod
    def _is_placeholder_key(cls, value: str | None) -> bool:
        key = (value or "").strip().lower()
        return key in cls._PLACEHOLDER_KEYS or "your_" in key or "your-" in key

    @classmethod
    def _credential_reason_for(cls, provider: str, api_key: str | None) -> str | None:
        """凭据判定的唯一实现:``None``=可用,否则稳定的 reason code。

        实例路径(``_mock_reason``)与 /health 的 ``classify_mode`` 共用它——
        模式分类只保留一份,不会出现"健康端点说 mock、实例说 live"的漂移。
        """
        if provider not in cls.SUPPORTED_PROVIDERS:
            return "unknown_provider"
        if provider == "qianfan":
            values = (config.qianfan_ak, config.qianfan_sk)
        elif provider == "zhipu":
            values = (api_key, config.zhipuai_api_key)
        else:
            values = (api_key,)
        if not any((value or "").strip() for value in values):
            return "missing_key"
        if any(cls._is_placeholder_key(value) for value in values):
            return "placeholder_key"
        return None

    def _credential_reason(self) -> str | None:
        return self._credential_reason_for(self.provider, self.api_key)

    @classmethod
    def config_warnings(cls) -> list[str]:
        """Provider↔key 形态问题的稳定码列表(W2.5:/health.llm_warnings)。

        只做静态配置检查(不发网络请求),与``classify_mode`` 的区别:
        分类回答"现在会以什么模式运行",这里回答"配置有什么形态问题"。
        码表稳定:qianfan-credentials-missing / zhipu-key-missing /
        remote-key-missing / provider-is-mock / mock-enabled /
        mock-allowed-in-production。
        """
        warnings: list[str] = []
        provider = (config.llm_provider or "").strip()
        api_key = (config.llm_api_key or "").strip()
        if provider == "qianfan":
            if not ((config.qianfan_ak or "").strip() and (config.qianfan_sk or "").strip()):
                warnings.append("qianfan-credentials-missing")
        elif provider == "zhipu":
            if not (api_key or (config.zhipuai_api_key or "").strip()):
                warnings.append("zhipu-key-missing")
        elif provider == "openai_compatible":
            local = cls._is_local_url(config.llm_api_base)
            if not local and (not api_key or cls._is_placeholder_key(api_key)):
                warnings.append("remote-key-missing")
        if provider == "mock":
            warnings.append("provider-is-mock")
        if cls._flag_enabled(os.getenv("MOCK_LLM")) or bool(config.mock_llm):
            warnings.append("mock-enabled")
        production = (getattr(config, "project_env", "") or "").strip().lower() in ("production", "prod")
        if production and bool(getattr(config, "allow_mock_in_prod", False)):
            warnings.append("mock-allowed-in-production")
        return warnings

    # ----------------------------------------------------------- sanitization

    @classmethod
    def _redact(cls, text: str) -> str:
        """Strip credentials from provider error messages.

        httpx exceptions embed the full request URL — for token-in-query
        providers (e.g. Qianfan ``?access_token=...``) that would leak the
        credential into chat responses and logs (CWE-209 / OWASP API8).
        """

        def _sub(match: re.Match) -> str:
            if match.group(1) is not None:
                return f"{match.group(1)}=***"
            if match.group(3) is not None:
                return f'"{match.group(3)}":"***"'
            matched = match.group(0)
            if matched.lower().startswith("bearer"):
                return "Bearer ***"
            if matched.lower().startswith("basic"):
                return "Basic ***"
            return "***"

        return _SECRET_PATTERN.sub(_sub, text)

    # ------------------------------------------------------------------ mock

    @staticmethod
    def _prompt_text(prompt: PromptInput) -> str:
        """把 str 或 message list 摊平成一段文本(仅 mock 摘要/日志用)。"""
        if isinstance(prompt, str):
            return prompt
        return "\n".join(str(message.get("content", "")) for message in prompt)

    def _mock_reply(self, prompt: PromptInput) -> str:
        # Extract user query part from multi-section prompt if present
        text = self._prompt_text(prompt)
        topic = text
        for marker in ("【患者主诉/提问】", "【患者主诉】", "【用户咨询】", "患者咨询：", "咨询问题：", "Question:"):
            if marker in text:
                topic = text.split(marker, 1)[1].strip()
                break
        # Strip ALL bracketed wrapper markers (untrusted-data delimiters etc.) and
        # collapse newlines so internal labels never leak into user-visible replies.
        topic = re.sub(r"【[^】]*】", "", topic)
        topic_preview = re.sub(r"\s+", " ", topic).strip()[:30]
        return (
            "【模拟诊疗意见】基于医学问诊信息，针对咨询内容"
            f"（{topic_preview}等）：建议完善血常规及影像学检查，遵医嘱治疗，注意生活调理。"
        )

    def _mock_stream(self, prompt: PromptInput) -> Iterator[str]:
        yield from self._mock_reply(prompt)

    # ------------------------------------------------- messages (W6.3 roles)

    @classmethod
    def _normalize_messages(cls, messages: PromptInput) -> list[dict[str, str]]:
        """str 旧路径逐字节保持;list 新路径 role 白名单 + 内容校验。"""
        if isinstance(messages, str):
            return [
                {"role": "system", "content": cls.SYSTEM_PREAMBLE},
                {"role": "user", "content": messages},
            ]
        if not isinstance(messages, Sequence):
            raise TypeError("prompt must be a str or a sequence of {'role', 'content'} mappings")
        if not messages:
            raise ValueError("prompt message list must contain at least one message")
        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"message {index} must be a mapping with 'role' and 'content'")
            role = str(message.get("role", "")).strip()
            if role not in cls.MESSAGE_ROLES:
                raise ValueError(
                    f"message {index} has unsupported role {role!r}; allowed roles: {sorted(cls.MESSAGE_ROLES)}"
                )
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"message {index} must carry non-empty string content")
            normalized.append({"role": role, "content": content})
        return normalized

    def _messages(self, prompt: PromptInput) -> list[dict[str, str]]:
        return self._normalize_messages(prompt)

    @staticmethod
    def _reject_unknown_kwargs(kwargs: Mapping[str, Any]) -> None:
        unknown = sorted(set(kwargs) - UnifiedLLM.KNOWN_INVOKE_KWARGS)
        if unknown:
            raise TypeError(f"invoke() got unexpected keyword argument(s): {', '.join(unknown)}")

    # ------------------------------------------------------------- providers

    def _bundle_openai(self, messages: PromptInput, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stream:
            payload["stream"] = True
        return {
            "url": f"{self.api_base.rstrip('/')}/chat/completions",
            "headers": {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            "payload": payload,
            "params": None,
        }

    def _qianfan_token(self) -> str:
        ak, sk = config.qianfan_ak or "", config.qianfan_sk or ""
        if not ak or not sk:
            raise RuntimeError("Qianfan AK/SK missing; set QIANFAN_AK / QIANFAN_SK")
        now = time.monotonic()
        if self._token_cache and now < self._token_cache[1]:
            return self._token_cache[0]
        # Single-flight refresh: concurrent first requests share one token fetch.
        with self._token_refresh_lock:
            now = time.monotonic()
            if self._token_cache and now < self._token_cache[1]:
                return self._token_cache[0]
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    "https://aip.baidubce.com/oauth/2.0/token",
                    params={"grant_type": "client_credentials", "client_id": ak, "client_secret": sk},
                )
                resp.raise_for_status()
                data = resp.json()
            token = data["access_token"]
            expires_in = max(float(data.get("expires_in", 1800)), 120.0)
            self._token_cache = (token, now + expires_in - 120.0)
            return token

    def _bundle_qianfan(self, messages: PromptInput, stream: bool) -> dict[str, Any]:
        model = self.model_name if self.model_name.startswith("ernie") else "ernie-4.0-8k"
        payload: dict[str, Any] = {
            "messages": self._messages(messages),
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "model": model,
        }
        if stream:
            payload["stream"] = True
        return {
            "url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro",
            "headers": {"Content-Type": "application/json"},
            "payload": payload,
            "params": {"access_token": self._qianfan_token()},
        }

    def _bundle_zhipu(self, messages: PromptInput, stream: bool) -> dict[str, Any]:
        api_key = self.api_key or config.zhipuai_api_key or ""
        if not api_key:
            raise RuntimeError("Zhipu API key missing; set ZHIPUAI_API_KEY")
        base = self.api_base if "bigmodel" in self.api_base else "https://open.bigmodel.cn/api/paas/v4"
        payload: dict[str, Any] = {
            "model": self.model_name if self.model_name.lower().startswith("glm") else "glm-4-flash",
            "messages": self._messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stream:
            payload["stream"] = True
        return {
            "url": f"{base.rstrip('/')}/chat/completions",
            "headers": {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            "payload": payload,
            "params": None,
        }

    def _bundle(self, messages: PromptInput, stream: bool) -> dict[str, Any]:
        if getattr(config, "llm_mode", "local") == "local" and self.enforce_mode and self.provider in {"qianfan", "zhipu"}:
            raise RuntimeError("remote LLM provider disabled by LLM_MODE=local")
        if getattr(config, "llm_mode", "local") == "local" and self.enforce_mode and self.provider == "openai_compatible" and not self._is_local_endpoint():
            raise RuntimeError("remote LLM endpoint disabled by LLM_MODE=local")
        # Unknown provider = fail closed: never silently fall back to the
        # OpenAI payload (auth header + endpoint would be wrong anyway).
        if self.provider not in self.SUPPORTED_PROVIDERS:
            raise RuntimeError(f"unsupported LLM provider {self.provider!r}")
        if self.provider == "qianfan":
            return self._bundle_qianfan(messages, stream)
        if self.provider == "zhipu":
            return self._bundle_zhipu(messages, stream)
        return self._bundle_openai(messages, stream)

    _LOCAL_HOSTS: ClassVar[frozenset[str]] = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "ollama"})

    @classmethod
    def _is_local_url(cls, api_base: str | None) -> bool:
        """Strict hostname allowlist per Python urllib.parse.urlsplit & OWASP SSRF guide.

        Substring matching on the raw URL let ``https://ollama.example.com/v1`` or
        ``http://remote.host/ollama`` masquerade as local endpoints.
        """
        raw = (api_base or "").strip()
        if not raw or "\\" in raw:
            return False
        try:
            parsed = urlsplit(raw)
        except ValueError:
            return False
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return False
        return (parsed.hostname or "") in cls._LOCAL_HOSTS

    def _is_local_endpoint(self) -> bool:
        return self._is_local_url(self.api_base)

    _PLACEHOLDER_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"your-api-key-here", "your_api_key", "changeme", "xxx", "sk-xxx", "none"}
    )

    def _mock_reason(self) -> str | None:
        if self.provider == "mock":
            return "provider_mock"
        if self._mock_enabled():
            return "mock_env"
        if self.provider == "openai_compatible" and self._is_local_endpoint():
            return None
        reason = self._credential_reason()
        if reason is not None:
            return reason
        return None

    def _production_allows_mock(self) -> bool:
        production = (getattr(config, "project_env", "") or "").strip().lower() in {"production", "prod"}
        return not production or bool(getattr(config, "allow_mock_in_prod", False))

    def _should_mock(self) -> bool:
        # 纯判定:不写实例状态(per-call 真值只在 LLMInvocation 上)。
        return self._mock_reason() is not None and self._production_allows_mock()

    @classmethod
    def classify_mode(cls, provider: str, api_key: str, api_base: str, mock_enabled: bool) -> tuple[str, str | None]:
        """Best-effort, non-network classification for /health; never guesses live on bad config."""
        if provider == "mock":
            return "mock", "provider_mock"
        if provider not in cls.SUPPORTED_PROVIDERS:
            return "error", "unknown_provider"
        local = cls._is_local_url(api_base)
        if mock_enabled:
            return "mock", "mock_env"
        if provider == "openai_compatible" and local:
            return "live", None
        reason = cls._credential_reason_for(provider, api_key)
        if reason is None:
            return "live", None
        production = (getattr(config, "project_env", "") or "").strip().lower() in {"production", "prod"}
        if production and not bool(getattr(config, "allow_mock_in_prod", False)):
            return "error", reason
        return "mock", reason

    def _effective_max_attempts(self) -> int:
        return max(1, self.max_retries if self.max_retries is not None else self.max_attempts)

    def _effective_backoff_base(self) -> float:
        return max(0.0, self.retry_backoff if self.retry_backoff is not None else self.backoff_base)

    # ------------------------------------------------------------------ core

    def stream(self, prompt: PromptInput, **kwargs: Any) -> Iterator[str]:
        """Development-only raw stream.

        Medical production responses must use the buffered ``invoke_with_status``
        path so AgentGraph reviewer, grounding, section-safety, and DosageGate
        run before any text is released. This method is retained only for
        backwards-compatible probes and refuses when PROJECT_ENV is production.
        """
        self._reject_unknown_kwargs(kwargs)
        if (getattr(config, "project_env", "") or "").strip().lower() in ("production", "prod"):
            raise RuntimeError("raw LLM streaming is disabled in production; use reviewed buffered responses")
        if self._should_mock():
            reason = self._mock_reason() or "mock_env"
            logger.warning("Raw LLM stream running in development/mock mode (reason={}); output is unreviewed.", reason)
            yield from self._mock_stream(prompt)
            return

        try:
            bundle = self._bundle(prompt, stream=True)
        except (RuntimeError, TypeError, ValueError):
            raise
        except Exception as exc:  # provider auth/network issues before starting
            logger.warning("Streaming degraded to mock ({}).", self._redact(str(exc)))
            yield from self._mock_stream(prompt)
            return

        has_yielded = False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST", bundle["url"], headers=bundle["headers"], json=bundle["payload"], params=bundle["params"]
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if chunk in ("", "[DONE]"):
                            continue
                        try:
                            data = json.loads(chunk)
                        except json.JSONDecodeError:
                            logger.debug("Skipping malformed SSE chunk: {}", self._redact(chunk[:200]))
                            continue
                        if "choices" in data:
                            choices = data.get("choices") or []
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            text = delta.get("content", "")
                        else:
                            text = data.get("result", "")
                        if text:
                            has_yielded = True
                            yield text
        except Exception as exc:
            redacted_err = self._redact(str(exc))
            if not has_yielded:
                # Safe to fall back to complete mock if no content has leaked yet
                logger.error("Streaming failed before output ({}); falling back to mock.", redacted_err)
                yield from self._mock_stream(prompt)
            else:
                # Already yielded real partial tokens — NEVER append a disconnected mock answer!
                logger.error("Streaming interrupted after yielding ({}); terminating cleanly.", redacted_err)
                yield f"\n\n【系统提示：流式传输已中断({redacted_err})，请刷新重试】"

    def invoke(self, prompt: PromptInput, **kwargs: Any) -> str:
        self._reject_unknown_kwargs(kwargs)
        return self.invoke_with_status(prompt, **kwargs).text

    def __call__(self, prompt: PromptInput, **kwargs: Any) -> str:
        return self.invoke(prompt, **kwargs)

    def invoke_with_status(
        self, prompt: PromptInput, stop: list[str] | None = None, **kwargs: Any
    ) -> LLMInvocation:
        """Generate with bounded retries; degrade to the mock when unconfigured.

        Returns an :class:`LLMInvocation` so the caller can read the outcome's
        status from the return value itself: the mock branch sets
        ``mocked=True``, exhausted retries set ``error_name`` to the exception
        type name, success leaves ``error_name=None``. Nothing per-call is stored
        on the instance — concurrent invocations cannot clear or inherit each
        other's status (only the aggregate ``mock_fallback_total`` diagnostic is).
        """
        self._reject_unknown_kwargs(kwargs)
        mock_reason = self._mock_reason()
        if self._should_mock():
            self._record_mock_fallback()
            logger.warning("LLM using structured mock fallback (reason={}).", mock_reason or "mock_env")
            return LLMInvocation(
                text=self._mock_reply(prompt),
                mocked=True,
                mode="mock",
                reason=mock_reason or "mock_env",
                diagnostic=mock_reason or "mock_env",
            )
        if mock_reason is not None and not self._production_allows_mock():
            # In production, an unconfigured remote provider is an error-mode
            # invocation, not a mock. Explicitly enabling mock in production
            # is the only exception.
            return LLMInvocation(
                text=("【系统错误提示】生产环境未配置可用的真实大模型凭据；"
                      "请检查 LLM_PROVIDER 与对应的 API Key。"),
                error_name="UnconfiguredInProduction",
                mode="error",
                reason=mock_reason,
                diagnostic=f"production_mock_blocked:{mock_reason}",
            )

        attempts = self._effective_max_attempts()
        backoff_base = self._effective_backoff_base()
        started = time.monotonic()
        deadline = started + self.retry_deadline
        last_error: Exception | None = None
        last_diagnostic: str | None = None
        for attempt in range(1, attempts + 1):
            if time.monotonic() >= deadline:
                last_error = TimeoutError("LLM retry deadline exceeded")
                last_diagnostic = "retry_deadline_exceeded"
                break
            try:
                bundle = self._bundle(prompt, stream=False)
                if stop:
                    bundle["payload"]["stop"] = stop
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        bundle["url"], headers=bundle["headers"], json=bundle["payload"], params=bundle["params"]
                    )
                    resp.raise_for_status()
                    data = resp.json()
                text = self._extract_text(data)
                if not text.strip():
                    # 空响应按可重试故障处理:受 max_attempts 与 retry_deadline
                    # 双重封顶;持续为空走下面的真实错误状态,绝不伪称成功。
                    raise _EmptyProviderResponseError("provider returned an empty response")
                return LLMInvocation(text=text, mode="live")
            except Exception as exc:
                last_error = exc
                retryable, diagnostic = self._classify_llm_error(exc)
                last_diagnostic = diagnostic
                logger.warning(
                    "LLM attempt {}/{} failed (retryable={}, reason={}): {}",
                    attempt, attempts, retryable, diagnostic, self._redact(str(exc)),
                )
                if not retryable:
                    break
                if attempt >= attempts:
                    break
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    last_error = TimeoutError("LLM retry deadline exceeded")
                    last_diagnostic = "retry_deadline_exceeded"
                    break
                # 指数全抖动;429 的 Retry-After 仍受 backoff_cap 封顶,且天花板
                # 永不超过本次调用的剩余时间预算(否则一个大的 Retry-After 就能
                # 把整个预算烧光,后面的重试再无机会)。
                ceiling = min(self.backoff_cap, backoff_base * (2 ** (attempt - 1)), remaining)
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                    retry_after = self._retry_after_seconds(exc.response.headers.get("Retry-After"))
                    if retry_after is not None:
                        ceiling = min(retry_after, self.backoff_cap, remaining)
                delay = random.uniform(0.0, max(0.0, ceiling))
                if delay >= remaining:
                    last_error = TimeoutError("LLM retry deadline exceeded")
                    last_diagnostic = "retry_deadline_exceeded"
                    break
                time.sleep(delay)
        if last_diagnostic == "empty_provider_response":
            # 有界重试后仍为空:真实错误状态(不是成功、不是 mock),原因码稳定。
            logger.error(
                "UnifiedLLM invocation failed (EmptyLLMResponse): provider returned "
                "an empty response on every attempt.",
            )
            return LLMInvocation(
                text="【系统错误提示】远程大模型返回空响应，请检查模型、请求参数与供应商服务状态。",
                error_name="EmptyLLMResponse",
                mode="error",
                diagnostic=last_diagnostic,
            )
        error_name = "RetryBudgetExceeded" if last_diagnostic == "retry_deadline_exceeded" else (
            type(last_error).__name__ if last_error else "unknown"
        )
        logger.error("UnifiedLLM invocation failed ({}): {}", error_name, self._redact(str(last_error)))
        diagnostic = last_diagnostic or "provider_error"
        safe_text = self._redact(str(last_error)) if last_error else "unknown error"
        return LLMInvocation(
            text=f"【系统错误提示】远程大模型调用失败({error_name})。{diagnostic}。请检查网络或 API Key。{safe_text}",
            error_name=error_name,
            mode="error",
            diagnostic=diagnostic,
        )

    @staticmethod
    def _extract_text(data: Any) -> str:
        if not isinstance(data, dict):
            return ""
        if "choices" in data:
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message") or {}
            content = message.get("content", "")
            return content if isinstance(content, str) else ""
        result = data.get("result", data.get("output", ""))
        return result if isinstance(result, str) else ""

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _classify_llm_error(exc: Exception) -> tuple[bool, str]:
        if isinstance(exc, _EmptyProviderResponseError):
            return True, "empty_provider_response"
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 401 or status == 403:
                return False, "check_LLM_API_KEY"
            if status in {408, 409, 429} or status >= 500:
                return True, f"http_{status}"
            if 400 <= status < 500:
                return False, "check_request_parameters"
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError, httpx.NetworkError)):
            return True, type(exc).__name__
        if isinstance(exc, json.JSONDecodeError):
            return True, "truncated_json_response"
        return False, "configuration_or_permanent_error"
