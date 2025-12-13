"""Small in-memory runtime metrics shim used for tests.

Provides the minimal API surface the application expects:
- `record_interaction(name, latency_ms=None)`
- `record_error(name, error_name)`
- `snapshot()` -> dict

This avoids pulling in the full metrics stack during development runs.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

_lock = threading.Lock()
_metrics: Dict[str, Any] = {
    "interactions": {},  # name -> {count, total_latency_ms}
    "errors": {},  # name -> count
    "last_updated": None,
}


def _now_ts() -> float:
    return time.time()


def record_interaction(name: str, latency_ms: float | None = None) -> None:
    """Record a single interaction metric (non-blocking, thread-safe)."""
    with _lock:
        entry = _metrics["interactions"].setdefault(
            name, {"count": 0, "total_latency_ms": 0.0}
        )
        entry["count"] += 1
        if latency_ms is not None:
            try:
                entry["total_latency_ms"] += float(latency_ms)
            except Exception:
                pass
        _metrics["last_updated"] = _now_ts()


def record_error(name: str, error_name: str | None = None) -> None:
    """Increment the error counter for the provided metric name."""
    with _lock:
        _metrics["errors"][name] = _metrics["errors"].get(name, 0) + 1
        _metrics["last_updated"] = _now_ts()


def snapshot() -> Dict[str, Any]:
    """Return a snapshot of current metrics (shallow copy)."""
    with _lock:
        # Build a compact, JSON-serializable snapshot
        interactions = {
            k: {
                "count": v["count"],
                "avg_latency_ms": (
                    (v["total_latency_ms"] / v["count"]) if v["count"] > 0 else None
                ),
            }
            for k, v in _metrics["interactions"].items()
        }
        return {
            "interactions": interactions,
            "errors": dict(_metrics["errors"]),
            "last_updated": _metrics["last_updated"],
        }


def get_stats() -> Dict[str, Any]:
    """Alias for `snapshot()` kept for compatibility."""
    return snapshot()
