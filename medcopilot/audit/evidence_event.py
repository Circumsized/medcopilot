"""Immutable, PHI-free evidence decision events for future audit adapters.

This module deliberately has no runtime integration.  Events are portable records
that can later be emitted to OpenTelemetry, immudb, or a WORM sink.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Mapping

GENESIS_HASH = "GENESIS"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9._:/-]+$")


def canonical_json(value: Any) -> str:
    """Return deterministic JSON (including stable Unicode and key ordering)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _check_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or not _ID.fullmatch(value):
        raise ValueError(f"{name} must be a non-empty PHI-free identifier")


def _check_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")


def _hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    event_id: str
    utc: str
    actor_id: str
    actor_role: str
    request_id: str
    session_id_hash: str
    policy_version: str
    policy_hash: str
    corpus_version: str
    evidence_ids: tuple[str, ...]
    model: str
    provider: str
    version: str
    input_hash: str
    output_hash: str
    gate_decision: str
    human_decision: Mapping[str, str] | None = None
    previous_hash: str = GENESIS_HASH
    event_hash: str = field(default="")

    def __post_init__(self) -> None:
        for name in ("event_id", "actor_id", "actor_role", "request_id", "corpus_version", "model", "provider", "version", "gate_decision"):
            _check_id(name, getattr(self, name))
        _check_hash("session_id_hash", self.session_id_hash)
        _check_hash("policy_hash", self.policy_hash)
        _check_hash("input_hash", self.input_hash)
        _check_hash("output_hash", self.output_hash)
        if self.previous_hash != GENESIS_HASH:
            _check_hash("previous_hash", self.previous_hash)
        try:
            parsed = datetime.fromisoformat(self.utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("utc must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None or not self.utc.endswith("Z"):
            raise ValueError("utc must be UTC and end with Z")
        if any(not isinstance(item, str) or not _ID.fullmatch(item) for item in self.evidence_ids):
            raise ValueError("evidence_ids must contain identifiers, never evidence text")
        if self.human_decision is not None:
            if set(self.human_decision) != {"decision", "actor_id", "actor_role"}:
                raise ValueError("human_decision requires decision, actor_id, and actor_role")
            for key in ("decision", "actor_id", "actor_role"):
                _check_id(f"human_decision.{key}", self.human_decision[key])
            if self.human_decision["actor_id"] != self.actor_id or self.human_decision["actor_role"] != self.actor_role:
                raise ValueError("human decision signer must match the event actor")
        if self.event_hash and not _HEX64.fullmatch(self.event_hash):
            raise ValueError("event_hash must be a lowercase SHA-256 hash")

    def payload(self) -> dict[str, Any]:
        data = {
            "event_id": self.event_id, "utc": self.utc, "actor_id": self.actor_id,
            "actor_role": self.actor_role, "request_id": self.request_id,
            "session_id_hash": self.session_id_hash, "policy_version": self.policy_version,
            "policy_hash": self.policy_hash, "corpus_version": self.corpus_version,
            "evidence_ids": list(self.evidence_ids), "model": self.model,
            "provider": self.provider, "version": self.version, "input_hash": self.input_hash,
            "output_hash": self.output_hash, "gate_decision": self.gate_decision,
            "human_decision": dict(self.human_decision) if self.human_decision else None,
            "previous_hash": self.previous_hash,
        }
        return data

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "event_hash": self.event_hash}

    def canonical(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def create(cls, **kwargs: Any) -> "EvidenceEvent":
        event = cls(**kwargs)
        return cls(**{**event.payload(), "event_hash": _hash(event.payload())})


@dataclass(frozen=True, slots=True)
class ChainVerification:
    ok: bool
    error: str = ""
    index: int = 0

    def __bool__(self) -> bool:
        return self.ok


def verify_chain(
    events: Iterable[EvidenceEvent],
    *,
    policy_version: str | None = None,
    policy_hash: str | None = None,
    evidence_ids: Iterable[str] | None = None,
) -> ChainVerification:
    """Verify order, hashes, signer consistency, and policy/evidence drift."""
    previous = GENESIS_HASH
    seen: set[str] = set()
    expected_evidence = set(evidence_ids) if evidence_ids is not None else None
    for index, event in enumerate(events, 1):
        if not isinstance(event, EvidenceEvent):
            return ChainVerification(False, "event is not EvidenceEvent", index)
        if event.event_id in seen:
            return ChainVerification(False, "duplicate event_id", index)
        seen.add(event.event_id)
        if event.previous_hash != previous:
            return ChainVerification(False, "out-of-order or broken previous_hash", index)
        if event.event_hash != _hash(event.payload()):
            return ChainVerification(False, "event_hash mismatch", index)
        if policy_version is not None and event.policy_version != policy_version:
            return ChainVerification(False, "policy version drift", index)
        if policy_hash is not None and event.policy_hash != policy_hash:
            return ChainVerification(False, "policy hash drift", index)
        if expected_evidence is not None and not set(event.evidence_ids).issubset(expected_evidence):
            return ChainVerification(False, "evidence drift", index)
        previous = event.event_hash
    return ChainVerification(True)
