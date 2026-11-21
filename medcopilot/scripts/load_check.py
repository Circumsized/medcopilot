"""轻量负载检查 CLI（Slice 7）。

启动真实 uvicorn 子进程（mock LLM），用并发 httpx 客户端打 /consultations，
报告 p50/p95 延迟与错误数。只做**连接层健康断言**（零错误 + 有界完成时间），
不做硬件相关的绝对吞吐阈值——吞吐结论必须在目标硬件上另行测量。

用法::

    python medcopilot/scripts/load_check.py [--concurrency 8] [--requests-per-worker 5] [--port 0]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx


def quantiles(values: list[float], probabilities: list[float]) -> list[float]:
    """线性插值分位数；空列表或越界概率即报错。"""
    if not values:
        raise ValueError("quantiles require at least one value")
    for p in probabilities:
        if not 0 <= p <= 1:
            raise ValueError(f"probability out of range: {p}")
    ordered = sorted(values)

    def at(p: float) -> float:
        position = p * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return [at(p) for p in probabilities]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ready(port: int, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.25)
    return False


def run_load_check(concurrency: int, requests_per_worker: int) -> dict[str, Any]:
    if concurrency < 1 or requests_per_worker < 1:
        raise ValueError("concurrency and requests_per_worker must be positive")
    port = _free_port()

    # The server is a loopback instance this script owns, so its environment is
    # pinned rather than inherited: an unset PROJECT_ENV is default-denied (503 on
    # every read) and the per-address rate limiter would throttle the load itself.
    # Both controls have their own contract tests; this check measures the
    # connection layer only.
    env = {
        **{k: v for k, v in os.environ.items() if k != "ADMIN_API_KEY"},
        "PYTHONPATH": ".",
        "MOCK_LLM": "1",
        "PROJECT_ENV": "development",
        "MED_RATE_LIMIT_ENABLED": "0",
        "RERANK_ENABLED": "0",
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "medcopilot.api.server:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=Path(__file__).resolve().parents[2], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    latencies: list[float] = []
    errors = 0
    stats_lock = threading.Lock()
    try:
        ready = _wait_ready(port)
        if not ready:
            raise RuntimeError("server did not become ready in time")
        total = concurrency * requests_per_worker

        def worker(worker_index: int) -> None:
            nonlocal errors
            with httpx.Client(timeout=30.0) as client:
                for i in range(requests_per_worker):
                    started = time.perf_counter()
                    failed = False
                    try:
                        response = client.post(
                            "http://127.0.0.1:" + str(port) + "/consultations",
                            json={"query": f"负载检查第 {worker_index}-{i} 问"},
                        )
                        if response.status_code != 200:
                            failed = True
                    except httpx.HTTPError:
                        failed = True
                    elapsed = time.perf_counter() - started
                    with stats_lock:
                        if failed:
                            errors += 1
                        latencies.append(elapsed)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(worker, range(concurrency)))

        p50, p95 = quantiles(latencies, [0.5, 0.95])
        return {
            "load_schema_version": 1,
            "ready": True,
            "concurrency": concurrency,
            "requests_per_worker": requests_per_worker,
            "total_requests": total,
            "errors": errors,
            "latency_p50_seconds": round(p50, 4),
            "latency_p95_seconds": round(p95, 4),
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="connection-layer load check with honest percentiles")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-per-worker", type=int, default=5)
    parser.add_argument("--output", default="outputs/load_report.json")
    args = parser.parse_args(argv)

    try:
        report = run_load_check(args.concurrency, args.requests_per_worker)
    except Exception as exc:
        print(f"FAIL: load check failed: {exc}", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"LOAD OK: total={report['total_requests']} errors={report['errors']} "
        f"p50={report['latency_p50_seconds']}s p95={report['latency_p95_seconds']}s report={out}"
    )
    return 0 if report["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
