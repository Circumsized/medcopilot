"""LLM 供应商连通性诊断（Slice 8）。

读取集中配置（或显式参数）对 provider 发起一次最小真实请求，输出结构化
诊断：可达性、鉴权、延迟。凭据绝不写入任何输出（诊断串统一脱敏）。

未配置（无 API key、mock 模式）时如实报告 ``not-configured``——探针不
猜测、不降级到 mock 冒充联调结果。退出码：0 探测成功 / 1 探测失败 /
2 未配置。真实密钥联调仍由持有密钥者执行；本工具让这一步一键化。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPORT_SCHEMA_VERSION = 1


def _redact(text: str, api_key: str | None) -> str:
    if api_key:
        text = text.replace(api_key, "***")
    return text


def probe(
    *,
    provider: str,
    api_base: str,
    api_key: str | None,
    model: str,
    timeout_seconds: float = 15.0,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """对 provider 发起一次最小 chat 请求并归类结果。

    ``transport`` 仅供测试注入；生产路径使用真实网络。
    """
    key = (api_key or "").strip()
    if provider == "mock" or not key:
        return {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "status": "not-configured",
            "provider": provider,
            "reason": "mock provider or empty API key; probe refuses to fake connectivity",
        }

    base = api_base.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }
    client_kwargs: dict[str, Any] = {"timeout": timeout_seconds}
    if transport is not None:
        client_kwargs["transport"] = transport

    started = time.perf_counter()
    try:
        with httpx.Client(**client_kwargs) as client:
            response = client.post(url, headers=headers, json=payload)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        return {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "status": "unreachable",
            "provider": provider,
            "endpoint": url,
            "error": _redact(str(exc), key),
        }
    latency = time.perf_counter() - started

    result: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "provider": provider,
        "endpoint": url,
        "model": model,
        "http_status": response.status_code,
        "latency_seconds": round(latency, 3),
    }
    if response.status_code == 200:
        result["status"] = "ok"
        return result
    if response.status_code in (401, 403):
        result["status"] = "auth-error"
    else:
        result["status"] = "http-error"
    result["error"] = _redact(response.text[:200], key)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM provider connectivity probe")
    parser.add_argument("--provider", default=None, help="default: LLM_PROVIDER")
    parser.add_argument("--api-base", default=None, help="default: LLM_API_BASE")
    parser.add_argument("--api-key", default=None, help="default: LLM_API_KEY (never printed)")
    parser.add_argument("--model", default=None, help="default: LLM_MODEL")
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--output", default="outputs/llm_probe.json")
    args = parser.parse_args(argv)

    from medcopilot.common.config import config

    provider = args.provider or config.llm_provider
    api_base = args.api_base or config.llm_api_base
    api_key = args.api_key if args.api_key is not None else (config.llm_api_key or "")
    model = args.model or config.llm_model
    mock_forced = config.mock_llm

    if mock_forced:
        result = probe(provider="mock", api_base=api_base, api_key="", model=model)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            "NOT-CONFIGURED: MOCK_LLM=1 forces mock mode; probe refuses to fake connectivity",
            file=sys.stderr,
        )
        return 2

    result = probe(
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        model=model,
        timeout_seconds=args.timeout_seconds,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    status = result["status"]
    print(f"LLM PROBE: status={status} provider={provider} endpoint={result.get('endpoint', '-')}")
    if "error" in result:
        print(f"  error: {result['error']}")
    if status == "not-configured":
        return 2
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
