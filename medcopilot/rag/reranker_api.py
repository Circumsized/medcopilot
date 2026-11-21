"""OpenAI-compatible reranking adapter with strict response validation."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import httpx

from medcopilot.common.config import config
from medcopilot.rag.clinical_chunker import Document

_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504})


class ApiReranker:
    """Call a configured ``/rerank`` endpoint without leaking response bodies."""

    def __init__(self, *, api_base: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float | None = None,
                 attempts: int = 3, transport: httpx.BaseTransport | None = None,
                 sleep=time.sleep) -> None:
        self.api_base = (api_base if api_base is not None else config.rerank_api_base).rstrip("/")
        self.api_key = (api_key if api_key is not None else config.rerank_api_key).strip()
        self.model = model or config.reranker_model
        self.timeout = float(timeout if timeout is not None else config.rerank_api_timeout)
        self.attempts = max(1, int(attempts))
        self._transport = transport
        self._sleep = sleep

    @property
    def endpoint(self) -> str:
        return f"{self.api_base}/rerank"

    def status(self) -> dict[str, Any]:
        return {"provider": "cloud", "model": self.model, "endpoint": self.endpoint,
                "api_key_configured": bool(self.api_key)}

    def rerank(self, query: str, documents: Sequence[Document], top_n: int = 3) -> list[tuple[Document, float]]:
        docs = list(documents)
        if not docs or top_n <= 0:
            return []
        payload = {"model": self.model, "query": query,
                   "documents": [doc.page_content for doc in docs], "top_n": min(top_n, len(docs))}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(1, self.attempts + 1):
            try:
                with httpx.Client(timeout=self.timeout, transport=self._transport) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                if attempt < self.attempts:
                    self._sleep(0.5 * attempt)
                    continue
                raise RuntimeError(f"rerank API unavailable ({type(exc).__name__})") from exc
            if response.status_code == 200:
                return self._parse(response, docs, top_n)
            if response.status_code in _RETRYABLE:
                if attempt < self.attempts:
                    self._sleep(0.5 * attempt)
                    continue
                raise RuntimeError(
                    f"rerank API unavailable after {self.attempts} attempts (last status {response.status_code})"
                )
            if response.status_code in (401, 403):
                raise RuntimeError("rerank API rejected the credential")
            raise RuntimeError(f"rerank API rejected the request (status {response.status_code})")
        raise RuntimeError("rerank API unavailable")

    @staticmethod
    def _parse(response: httpx.Response, docs: list[Document], top_n: int) -> list[tuple[Document, float]]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("rerank API returned a non-JSON body") from exc
        data = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise RuntimeError("rerank API response has no 'results' array")
        seen: set[int] = set()
        scored_with_index: list[tuple[Document, float, int]] = []
        for row in data:
            if not isinstance(row, dict) or type(row.get("index")) is not int:
                raise RuntimeError("rerank API result has invalid index")
            index = row["index"]
            score = row.get("relevance_score", row.get("score"))
            if index < 0 or index >= len(docs) or index in seen or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise RuntimeError("rerank API result has invalid index or score")
            seen.add(index)
            scored_with_index.append((docs[index], float(score), index))
        expected = min(top_n, len(docs))
        if len(scored_with_index) < expected:
            raise RuntimeError(f"rerank API returned {len(scored_with_index)} results, expected at least {expected}")
        scored_with_index.sort(key=lambda item: (item[1], -item[2]), reverse=True)
        return [(doc, score) for doc, score, _ in scored_with_index[:top_n]]


def build_api_reranker() -> ApiReranker | None:
    if config.rerank_provider != "cloud":
        return None
    if not config.allow_text_egress:
        raise RuntimeError("RERANK_PROVIDER=cloud requires ALLOW_TEXT_EGRESS=true")
    if not config.rerank_api_base or not config.rerank_api_key:
        raise RuntimeError("RERANK_PROVIDER=cloud requires RERANK_API_BASE and RERANK_API_KEY")
    return ApiReranker()
