"""Bounded, thread-safe in-process consultation session store.

Single-worker semantics by design (ingestion and session state stay
process-local). The store is an LRU with a sliding TTL:

* Touching a session (create or lookup) refreshes its last-access clock.
* When capacity is reached, the least-recently-used **and not in use** entry is
  evicted; entries whose workflow call is still running are protected.
* If every entry is in use, creation fails with ``SessionStoreFull`` so the API
  can answer 503 instead of growing beyond the configured bound.
* Lookups of unknown / expired / evicted ids return ``None`` — the API layer
  maps that to 404 for ordinary requests.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable

from medcopilot.workflow.agent_graph import Session

SESSION_ID_MAX_CHARS = 128


class SessionStoreFull(Exception):
    """Raised when capacity is full and every entry is currently in use."""


class SessionStore:
    def __init__(
        self,
        *,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float],
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._lock = threading.Lock()
        # session_id -> (session, last_access, in_use_count)
        self._entries: OrderedDict[str, tuple[Session, float, int]] = OrderedDict()
        # Per-session serialization locks; created/lifetime-managed together with
        # the entry so evicted sessions cannot leak locks without bound.
        self._locks: dict[str, threading.Lock] = {}

    # ---------------------------------------------------------------- helpers

    def new_session_id(self) -> str:
        """Unpredictable opaque id; caller-provided values are never reused."""
        return uuid.uuid4().hex

    @staticmethod
    def normalize_session_id(raw: str | None) -> str | None:
        """Blank => not provided; >128 chars is a 422 concern for the API layer."""
        if raw is None:
            return None
        trimmed = raw.strip()
        if not trimmed:
            return None
        return trimmed

    def validate_session_id(self, raw: str | None) -> str | None:
        """Normalized id, or raise ValueError when over the length contract."""
        normalized = self.normalize_session_id(raw)
        if normalized is not None and len(normalized) > SESSION_ID_MAX_CHARS:
            raise ValueError("session_id too long")
        return normalized

    # ----------------------------------------------------------------- public

    def lock_for(self, session_id: str) -> threading.Lock:
        """Serialization lock for an acquired/created session.

        Must be called only after :meth:`acquire` returned a session or
        :meth:`create` succeeded: while the entry is in use it cannot be
        evicted, so the lock is guaranteed to still exist.
        """
        with self._lock:
            return self._locks.setdefault(session_id, threading.Lock())

    def create(self) -> tuple[str, Session]:
        """Create a fresh session, evicting an idle LRU entry if needed."""
        with self._lock:
            self._evict_expired_locked()
            if len(self._entries) >= self._max_entries:
                evicted = self._evict_one_idle_locked()
                if not evicted:
                    raise SessionStoreFull
            session_id = self.new_session_id()
            while session_id in self._entries:  # pragma: no cover - uuid4 space
                session_id = self.new_session_id()
            self._entries[session_id] = (Session(), self._clock(), 1)
            self._locks.setdefault(session_id, threading.Lock())
            return session_id, self._entries[session_id][0]

    @property
    def size(self) -> int:
        """Live entry count (observability only: /health + /metrics gauges).

        Read-only snapshot under the store lock; never used for control flow —
        capacity decisions stay inside :meth:`create`.
        """
        with self._lock:
            return len(self._entries)

    def acquire(self, session_id: str) -> Session | None:
        """Return the live session for an explicit id and mark it in use.

        Unknown, expired or evicted ids yield ``None``. While marked in use the
        entry cannot be evicted; callers MUST release in a finally block.
        """
        with self._lock:
            self._evict_expired_locked()
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            session, _, in_use = entry
            self._entries.move_to_end(session_id)
            self._entries[session_id] = (session, self._clock(), in_use + 1)
            return session

    def release(self, session_id: str) -> None:
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return
            session, last_access, in_use = entry
            self._entries[session_id] = (session, last_access, max(0, in_use - 1))

    @contextmanager
    def session_lock(self, session_id: str) -> Iterator[threading.Lock]:
        """Acquire the per-session lock and always release the entry on exit.

        Only valid after :meth:`acquire` returned a session or :meth:`create`
        succeeded — the in-use mark taken there guarantees the lock still
        exists and the entry cannot be evicted while held. Replaces the
        hand-rolled ``try/finally`` whose single missed ``release`` made an
        entry permanently un-evictable.
        """
        lock = self.lock_for(session_id)
        with lock:  # actually hold the critical section, not merely fetch it
            try:
                yield lock
            finally:
                self.release(session_id)

    # -------------------------------------------------------------- internals

    def _evict_expired_locked(self) -> None:
        now = self._clock()
        expired = [
            sid
            for sid, (_, last_access, in_use) in self._entries.items()
            if in_use == 0 and now - last_access >= self._ttl
        ]
        for sid in expired:
            del self._entries[sid]
            self._locks.pop(sid, None)

    def _evict_one_idle_locked(self) -> bool:
        """Evict the least recently used entry that is not in use."""
        for sid, (_, _, in_use) in self._entries.items():
            if in_use == 0:
                del self._entries[sid]
                self._locks.pop(sid, None)
                return True
        return False
