"""Serving snapshot holder: one immutable runtime generation per request.

Replaces the previous mutable ``state: dict[str, Any]`` whose keys were
written one-by-one during ingestion/rebuild — a concurrent reader could
observe a torn half-swapped generation (new retriever wired to old
documents). Here the whole object set is published with a single reference
write (``ServingHolder.swap``), so any reader either sees the previous
generation in full or the new one in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request

from medcopilot.api.session_store import SessionStore


@dataclass(frozen=True)
class AppRuntime:
    """冻结快照：读取方取一次引用即看到一致的对象集。

    字段即旧 ``state`` dict 的全部键：documents（权威语料快照）、retriever、
    dense_encoder、ocr、workflow（AgentGraph）、audit（审计 sink）。ingest /
    rebuild 通过 :func:`commit_serving` 以单引用写发布新一代快照；并发请求
    持有的旧引用保持完整可用，不会撕裂也不会被原地改写。
    """

    documents: list[Any]
    retriever: Any
    dense_encoder: Any
    ocr: Any
    workflow: Any
    audit: Any


class ServingHolder:
    """Holds the current :class:`AppRuntime`; ``swap`` is the commit point."""

    def __init__(self, runtime: AppRuntime) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> AppRuntime:
        return self._runtime

    def swap(self, runtime: AppRuntime) -> None:
        # Single reference write = atomic publication under the GIL; readers
        # that already hold the old AppRuntime keep a coherent generation.
        self._runtime = runtime


def get_runtime(request: Request) -> AppRuntime:
    """FastAPI dependency: the request's serving snapshot."""
    return request.app.state.runtime_holder.runtime


def get_sessions(request: Request) -> SessionStore:
    """FastAPI dependency: the process-local session store."""
    return request.app.state.sessions


def commit_serving(
    holder: ServingHolder,
    *,
    documents: list[Any],
    retriever: Any,
    dense_encoder: Any,
    ocr: Any,
    workflow: Any,
    audit: Any,
) -> AppRuntime:
    """Publish a new serving generation (ingest/rebuild 的唯一 serving 提交点).

    须在 ``ingest_lock`` 内调用。先让图看到新检索器（consultation 的检索读
    self.workflow.retriever，该赋值是 serving 切换的提交点），再整体构建
    AppRuntime 并 ``holder.swap()``——保持既有"先建后交"语义：所有新对象在
    提交前都已构造完毕，提交本身不再有可失败步骤。
    """
    workflow.retriever = retriever
    runtime = AppRuntime(
        documents=documents,
        retriever=retriever,
        dense_encoder=dense_encoder,
        ocr=ocr,
        workflow=workflow,
        audit=audit,
    )
    holder.swap(runtime)
    return runtime
