"""Optional offline adapters for official RAG evaluation frameworks.

This module is deliberately outside the serving graph.  It never changes an
``AgentGraph`` result or any safety gate.  The adapter only turns an already
completed consultation into a source-scoped evaluation sample and, when an
optional framework is installed, provides a stable score envelope.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

if TYPE_CHECKING:
    from medcopilot.workflow.agent_graph import ConsultationResult

PROTOCOL_VERSION = "rag_official_adapter_v1"
_SUPPORTED = {"deepeval", "ragas"}


@dataclass(frozen=True)
class EvaluationSample:
    """Framework-neutral sample with contexts restricted to selected sources."""

    user_input: str
    response: str
    retrieved_contexts: tuple[str, ...]
    reference: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _source_contexts(result: Mapping[str, Any], contexts: Mapping[str, str]) -> tuple[str, ...]:
    """Return only caller-supplied text for source IDs served by the result.

    ``sources`` contains untrusted metadata.  In particular, source-provided
    text is never accepted as evidence.  The caller must provide the local
    chunk store separately, keyed by ``chunk_id``.
    """
    selected: list[str] = []
    seen: set[str] = set()
    for source in result.get("sources") or []:
        if not isinstance(source, Mapping):
            continue
        chunk_id = source.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id in contexts and chunk_id not in seen:
            text = contexts[chunk_id]
            if isinstance(text, str) and text:
                selected.append(text)
                seen.add(chunk_id)
    return tuple(selected)


def sample_from_consultation(
    query: str,
    result: ConsultationResult | Mapping[str, Any],
    *,
    source_contexts: Mapping[str, str],
    reference: str | None = None,
) -> EvaluationSample:
    """Map a ``ConsultationResult`` (or compatible mapping) to a sample.

    ``source_contexts`` is intentionally explicit: a result's source metadata
    identifies allowed chunks, but does not grant permission to treat metadata
    or OCR/user text as clinical evidence.
    """
    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    if not isinstance(result, Mapping):
        raise TypeError("result must be a ConsultationResult-compatible mapping")
    contexts = _source_contexts(result, source_contexts)
    return EvaluationSample(
        user_input=query,
        response=str(result.get("answer", "")),
        retrieved_contexts=contexts,
        reference=reference,
        metadata={
            "intent": result.get("intent"),
            "abstained": bool(result.get("abstained", False)),
            "is_emergency": bool(result.get("is_emergency", False)),
            "policy_version": result.get("policy_version"),
            "source_count": len(contexts),
            "source_sections": tuple(
                source.get("section")
                for source in (result.get("sources") or [])
                if isinstance(source, Mapping) and isinstance(source.get("section"), str)
            ),
        },
    )


class OfficialRAGAdapter:
    """A dependency-optional, network-free adapter facade.

    ``runner`` is injectable for local framework integrations and tests.  It
    receives ``EvaluationSample`` objects and returns metric-name/score pairs.
    Production serving code never needs to construct this class.
    """

    def __init__(
        self,
        provider: str,
        *,
        offline_only: bool = True,
        runner: Callable[[list[EvaluationSample]], Mapping[str, float]] | None = None,
    ) -> None:
        provider = provider.strip().lower()
        if provider not in _SUPPORTED:
            raise ValueError(f"provider must be one of {sorted(_SUPPORTED)}")
        if not offline_only:
            raise ValueError("offline_only must be True; remote evaluation is not permitted")
        self.provider = provider
        self.offline_only = True
        self.runner = runner

    @property
    def dependency_available(self) -> bool:
        return find_spec(self.provider) is not None

    def evaluate(self, samples: Iterable[EvaluationSample]) -> dict[str, Any]:
        samples = list(samples)
        envelope: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "adapter": self.provider,
            "offline_only": True,
            "status": "not-configured",
            "sample_count": len(samples),
            "scores": {},
            "error": None,
        }
        if not samples:
            envelope["error"] = "no-samples"
            return envelope
        if self.runner is None:
            if not self.dependency_available:
                envelope["error"] = f"optional dependency not installed: {self.provider}"
                return envelope
            envelope["error"] = "no offline runner configured"
            return envelope
        try:
            raw = self.runner(samples)
            if not isinstance(raw, Mapping):
                raise TypeError("runner must return a mapping of metric scores")
            scores = {str(k): float(v) for k, v in raw.items()}
            if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in scores.values()):
                raise ValueError("metric scores must be finite numbers between 0 and 1")
            envelope["scores"] = scores
            envelope["status"] = "ok"
        except Exception as exc:  # adapter boundary must not affect serving
            envelope["error"] = f"{type(exc).__name__}: {exc}"
        return envelope


def evaluate_consultations(
    provider: str,
    consultations: Iterable[tuple[str, Mapping[str, Any], Mapping[str, str]]],
    *,
    offline_only: bool = True,
    runner: Callable[[list[EvaluationSample]], Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Convenience API for ``(query, ConsultationResult, local_contexts)`` rows."""
    samples = [
        sample_from_consultation(query, result, source_contexts=contexts)
        for query, result, contexts in consultations
    ]
    return OfficialRAGAdapter(provider, offline_only=offline_only, runner=runner).evaluate(samples)
