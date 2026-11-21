"""Real provider matrix via UnifiedLLM; 0=ok, 1=failed, 2=not-configured.

Credentials come from AppConfig/.env. No endpoint, reply or credential is
included in JSON. Mock mode is never used as connectivity evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPORT_SCHEMA_VERSION = 1
PROVIDERS = ("openai_compatible", "qianfan", "zhipu")


def probe_provider(*, provider: str, config: Any, timeout_seconds: float = 15.0) -> dict[str, Any]:
    """One real call using the adapter's per-invocation status contract."""
    result: dict[str, Any] = {"provider": provider}
    if provider not in PROVIDERS:
        return {"provider": "unsupported", "status": "not-configured", "reason": "unsupported provider"}
    try:
        from medcopilot.llm.unified_llm import UnifiedLLM
    except ImportError:
        return {**result, "status": "not-configured", "reason": "missing llm-probe dependencies"}

    def usable(value: str | None) -> bool:
        key = (value or "").strip()
        return bool(key) and key.lower() not in UnifiedLLM._PLACEHOLDER_KEYS and not key.lower().startswith("your_")

    key = config.llm_api_key
    if provider == "zhipu":
        key = config.zhipuai_api_key or key
    elif provider == "qianfan":
        # Adapter's mock guard needs a usable api_key even though Qianfan's
        # real authentication uses AppConfig AK/SK inside _qianfan_token().
        key = config.qianfan_ak
        if not usable(config.qianfan_sk):
            return {**result, "status": "not-configured", "reason": "missing Qianfan credentials"}
    if config.mock_llm or not usable(key):
        return {**result, "status": "not-configured", "reason": "mock mode or missing usable credentials"}

    from medcopilot.common.logger import logger

    started = time.perf_counter()
    # Provider exceptions can contain URLs or arbitrary secrets. Do not emit
    # their text, including the adapter's logging, to diagnostic artifacts.
    logger.disable("medcopilot.llm.unified_llm")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            llm = UnifiedLLM(
                provider=provider, api_key=key, api_base=config.llm_api_base,
                model_name=config.llm_model, timeout=timeout_seconds,
                max_tokens=8, max_retries=1, retry_backoff=0,
            )
            if llm._should_mock():
                return {**result, "status": "not-configured", "reason": "adapter mock mode enabled"}
            outcome = llm.invoke_with_status("Reply only with OK.")
        if outcome.mocked:
            result.update(status="not-configured", reason="adapter returned mock; not connectivity evidence")
        elif outcome.error_name:
            result.update(status="error", reason="provider invocation failed")
        elif not isinstance(outcome.text, str) or not outcome.text.strip():
            result.update(status="error", reason="empty provider response")
        else:
            result["status"] = "ok"
    except Exception:
        result.update(status="error", reason="adapter invocation failed")
    finally:
        logger.enable("medcopilot.llm.unified_llm")
    result["latency_seconds"] = round(time.perf_counter() - started, 3)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", default=",".join(PROVIDERS), help="comma-separated provider matrix")
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--output", default=None, help="optional JSON report path; JSON always goes to stdout")
    args = parser.parse_args(argv)
    providers = list(dict.fromkeys(item.strip() for item in args.providers.split(",") if item.strip()))
    if not providers or any(name not in PROVIDERS for name in providers):
        parser.error("--providers must contain supported provider names")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    try:
        from medcopilot.common.config import config

        results = [probe_provider(provider=name, config=config, timeout_seconds=args.timeout_seconds) for name in providers]
    except ImportError:
        results = [{"provider": name, "status": "not-configured", "reason": "missing dependencies"} for name in providers]
    except Exception:
        results = [{"provider": name, "status": "not-configured", "reason": "invalid configuration"} for name in providers]
    statuses = {item["status"] for item in results}
    status = "error" if "error" in statuses else "not-configured" if "not-configured" in statuses else "ok"
    report = {"report_schema_version": REPORT_SCHEMA_VERSION, "status": status, "results": results}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        try:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded + "\n", encoding="utf-8")
        except OSError:
            print(json.dumps({"status": "error", "reason": "report write failed"}))
            return 1
    print(encoded)
    return {"ok": 0, "error": 1, "not-configured": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
