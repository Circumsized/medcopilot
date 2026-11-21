"""Dependency-free, process-local abuse controls for the API (OWASP LLM10 / API4).

Scope: DoS and cost control only. These primitives slow a client down and cap
in-flight work; they are NOT a content-safety boundary (best-of-N probing is
merely slowed), and being process-local they assume the documented
``--workers 1`` deployment.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int, limit: int):
        self.retry_after = max(1, retry_after)
        self.limit = limit
        super().__init__("rate limit exceeded")


class ConcurrencyLimitExceeded(Exception):
    def __init__(self, limit: int):
        self.limit = limit
        super().__init__("too many in-flight expensive requests")


class SlidingWindowLimiter:
    """Per-key sliding-window limiter; safe for the single-worker deployment."""

    # ponytail: draining keys are swept when the table is full, but that sweep
    # only reclaims DRAINED keys — a flood of fresh identities (rotating source
    # IPs in no-key dev mode) keeps every slot busy, so the table would still
    # grow without bound. The hard bound below rejects an unseen key once the
    # table is full after sweeping, trading a (temporary) 429 for a bounded
    # memory footprint. Production keys by the single API-key identity, so the
    # bound is unreachable there; the 429 path exists for the no-key case.
    # Upgrade path if per-IP quotas ever matter: an LRU/TTL map.
    MAX_KEYS = 4096

    def __init__(self, limit: int, window_seconds: float = 60.0, clock=time.monotonic):
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("limit and window_seconds must be positive")
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._events = defaultdict(deque)
        self.rejected = 0

    def check(self, key: str) -> None:
        now = self._clock()
        with self._lock:
            cutoff = now - self.window_seconds
            # Sweep only when the table is at capacity: reclaims drained keys.
            if len(self._events) >= self.MAX_KEYS:
                for stale in [k for k, v in self._events.items() if not v or v[-1] <= cutoff]:
                    del self._events[stale]
            events = self._events.get(key)
            if events is None:
                # Hard bound: sweeping freed nothing and the table is still
                # full ⇒ every slot is held by a fresh, in-window identity.
                # Fail closed (429) instead of growing the table without limit.
                if len(self._events) >= self.MAX_KEYS:
                    self.rejected += 1
                    raise RateLimitExceeded(int(self.window_seconds), self.limit)
                events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                self.rejected += 1
                retry_after = int(max(1, events[0] + self.window_seconds - now + 0.999))
                raise RateLimitExceeded(retry_after, self.limit)
            events.append(now)


class ConcurrencyGuard:
    """Bounded in-flight counter for expensive paths (retrieval + rerank + LLM + OCR).

    Acquisition never blocks: a full guard rejects instantly instead of parking
    a worker thread. Parked threads are what let slow or same-session requests
    exhaust the shared thread pool and starve emergency triage and ``/health``.

    Every expensive path holds one slot for its whole duration — including the
    admin ingest/rebuild paths while they PARK on ``ingest_lock`` (see
    :meth:`waiter`): lock waiters therefore consume the same bounded budget,
    so a queue of stalled rebuilds cannot starve the shared thread pool.
    """

    def __init__(self, limit: int):
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit
        self._semaphore = threading.BoundedSemaphore(limit)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.rejected = 0
        # Threads parked on / inside the ingest-rebuild critical region while
        # holding a slot. Bounded by construction (a waiter owns a slot), and
        # observable so lock congestion is visible before it becomes starvation.
        self.waiters = 0

    @contextmanager
    def slot(self) -> Iterator[None]:
        if not self._semaphore.acquire(blocking=False):
            with self._lock:
                self.rejected += 1
            raise ConcurrencyLimitExceeded(self.limit)
        with self._lock:
            self.in_flight += 1
        try:
            yield
        finally:
            with self._lock:
                self.in_flight -= 1
            self._semaphore.release()

    @contextmanager
    def waiter(self) -> Iterator[None]:
        """Mark a slot-holding thread that is queued on the ingest/rebuild lock."""
        with self._lock:
            self.waiters += 1
        try:
            yield
        finally:
            with self._lock:
                self.waiters -= 1
