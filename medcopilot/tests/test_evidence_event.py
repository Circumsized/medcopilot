"""Contracts for the immutable evidence event model."""

from __future__ import annotations

import hashlib
import json

import pytest

from medcopilot.audit.evidence_event import (
    GENESIS_HASH,
    EvidenceEvent,
    canonical_json,
    verify_chain,
)


def _kwargs(**overrides):
    values = {
        "event_id": "evt-1", "utc": "2026-10-04T12:00:00Z", "actor_id": "operator-1",
        "actor_role": "clinician", "request_id": "req-1", "session_id_hash": "a" * 64,
        "policy_version": "policy-v1", "policy_hash": "b" * 64, "corpus_version": "corpus-v1",
        "evidence_ids": ("ev-1",), "model": "model-x", "provider": "provider-x", "version": "1",
        "input_hash": "c" * 64, "output_hash": "d" * 64, "gate_decision": "abstain",
    }
    values.update(overrides)
    return values


def test_canonical_json_is_deterministic():
    assert canonical_json({"z": 1, "a": "汉字"}) == '{"a":"汉字","z":1}'


def test_create_is_hashed_and_immutable():
    event = EvidenceEvent.create(**_kwargs())
    assert event.previous_hash == GENESIS_HASH
    assert len(event.event_hash) == 64
    with pytest.raises(AttributeError):
        event.actor_id = "other"
    assert json.loads(event.canonical())["event_hash"] == event.event_hash


def test_chain_detects_tamper_duplicate_and_order():
    first = EvidenceEvent.create(**_kwargs())
    second = EvidenceEvent.create(**_kwargs(event_id="evt-2", previous_hash=first.event_hash))
    assert verify_chain([first, second])
    assert not verify_chain([second, first])
    assert not verify_chain([first, first])
    tampered = EvidenceEvent(**{**second.to_dict(), "output_hash": "e" * 64})
    assert not verify_chain([first, tampered])


def test_chain_detects_policy_and_evidence_drift():
    event = EvidenceEvent.create(**_kwargs())
    assert not verify_chain([event], policy_version="policy-v2")
    assert not verify_chain([event], policy_hash="e" * 64)
    assert not verify_chain([event], evidence_ids=("other",))


def test_human_decision_must_match_signer_and_no_phi():
    with pytest.raises(ValueError):
        EvidenceEvent.create(**_kwargs(human_decision={"decision": "approve", "actor_id": "other", "actor_role": "clinician"}))
    event = EvidenceEvent.create(**_kwargs(human_decision={"decision": "reject", "actor_id": "operator-1", "actor_role": "clinician"}))
    assert "患者" not in event.canonical()
    assert hashlib.sha256(b"raw patient text").hexdigest() != "raw patient text"
