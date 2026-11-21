"""Zero-dependency Prometheus metrics registry (W5.6).

No exposition library: counters / gauges / a fixed-bucket histogram with a
thread-safe registry and a ``text/plain; version=0.0.4`` exporter, so behaviour
is CI-testable by string assertion and the dependency surface stays at zero.
The interface mirrors the small subset of ``prometheus_client`` we need
(``inc`` / ``set`` / ``observe`` / ``export``) so a later swap is mechanical.

Label discipline (hard rule, parent-coordinated): metrics carry ONLY
non-sensitive labels — endpoint, status, component, mode, intent. Never
query text, host, client IP, session id or key material. Label cardinality is
hard-bounded (``MAX_SERIES`` per metric name): a flood of distinct label
values must not grow the registry without limit, and once the bound is hit new
series are dropped and counted in ``med_rag_dropped_series_total`` rather
than admitted.
"""

from __future__ import annotations

import threading
from typing import Any

#: Fixed buckets (seconds) — covers the fast in-memory path to a hung upstream.
BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
MAX_SERIES = 512

SeriesKey = tuple[str, tuple[tuple[str, str], ...]]


def _label_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _format_value(value: float) -> str:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):  # NaN/Inf never exported
        return "0"
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return repr(number)


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{k}="{_escape_label(v)}"' for k, v in labels) + "}"


class MetricsRegistry:
    """Thread-safe counter/gauge/histogram registry with a Prom exporter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[SeriesKey, float] = {}
        self._gauges: dict[SeriesKey, float] = {}
        self._histograms: dict[SeriesKey, dict[str, Any]] = {}
        self._dropped_series = 0

    # ------------------------------------------------------------- ingestion

    def inc(self, name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:
        key: SeriesKey = (name, _label_key(labels))
        with self._lock:
            if not self._admit(key, self._counters):
                return
            self._counters[key] = self._counters.get(key, 0.0) + value

    def set(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key: SeriesKey = (name, _label_key(labels))
        with self._lock:
            if not self._admit(key, self._gauges):
                return
            self._gauges[key] = float(value)

    def set_counter(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Absolute set for a monotonic counter read off an external observable.

        Keeps the Prometheus counter contract (never decreasing within a
        process) when the source of truth lives elsewhere, e.g. the audit
        sink's cumulative write-failure count.
        """
        key: SeriesKey = (name, _label_key(labels))
        with self._lock:
            if not self._admit(key, self._counters):
                return
            self._counters[key] = max(self._counters.get(key, 0.0), float(value))

    def observe(self, name: str, seconds: float, labels: dict[str, str] | None = None) -> None:
        key: SeriesKey = (name, _label_key(labels))
        with self._lock:
            if not self._admit(key, self._histograms, histogram=True):
                return
            entry = self._histograms[key]
            entry["count"] += 1
            entry["sum"] += seconds
            for index, upper in enumerate(BUCKETS):
                if seconds <= upper:
                    entry["buckets"][index] += 1

    def _admit(self, key: SeriesKey, store: dict, histogram: bool = False) -> bool:
        """Register a new series, or drop it (counted) when the bound is hit."""
        if key in store:
            return True
        if len(store) >= MAX_SERIES:
            self._dropped_series += 1
            return False
        if histogram:
            store[key] = {"count": 0, "sum": 0.0, "buckets": [0] * len(BUCKETS)}
        else:
            store[key] = 0.0
        return True

    # ------------------------------------------------------------- exposition

    def export_prom(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{_render_labels(labels)} {_format_value(value)}")
            for (name, labels), value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{_render_labels(labels)} {_format_value(value)}")
            for (name, labels), entry in sorted(self._histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                for index, upper in enumerate(BUCKETS):
                    rendered = _render_labels(labels + (("le", _format_value(upper)),))
                    lines.append(f"{name}_bucket{rendered} {_format_value(entry['buckets'][index])}")
                rendered_inf = _render_labels(labels + (("le", "+Inf"),))
                lines.append(f"{name}_bucket{rendered_inf} {_format_value(entry['count'])}")
                lines.append(f"{name}_sum{_render_labels(labels)} {_format_value(entry['sum'])}")
                lines.append(f"{name}_count{_render_labels(labels)} {_format_value(entry['count'])}")
            if self._dropped_series:
                lines.append("# TYPE med_rag_dropped_series_total counter")
                lines.append(f"med_rag_dropped_series_total {_format_value(self._dropped_series)}")
        return "\n".join(lines) + "\n" if lines else ""

    # ------------------------------------------------------------------ misc

    def reset(self) -> None:
        """Test/maintenance hook: drop every series."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._dropped_series = 0

    @property
    def dropped_series(self) -> int:
        with self._lock:
            return self._dropped_series
