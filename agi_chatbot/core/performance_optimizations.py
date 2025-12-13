"""Shim for performance optimizations compatibility.

Provides `PerformanceMonitor`, `monitor_performance`, and `get_performance_stats`.
"""
from typing import Dict, Any
from functools import wraps
try:
    from agi_chatbot.cache_manager import EnhancedCache
except Exception:
    EnhancedCache = None


class PerformanceMonitor:
    def __init__(self) -> None:
        self._stats: Dict[str, Any] = {"requests": 0}

    def record(self) -> None:
        self._stats["requests"] += 1

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)


monitor = PerformanceMonitor()


def monitor_performance(name=None):
    """Decorator factory and accessor for the performance monitor.

    Usage patterns supported by the shim:
    - `@monitor_performance("health")` -> returns a decorator that wraps the function
      (no-op for shim, but records a metric).
    - `monitor_performance()` -> returns the monitor instance.
    - `@monitor_performance` (used without parentheses) -> supported as simple decorator.
    """
    # If used as `@monitor_performance` (no args), name will be the function.
    if callable(name):
        func = name

        @wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                monitor.record()
            except Exception:
                pass
            return func(*args, **kwargs)

        return _wrapper

    # If used as `@monitor_performance("tag")` return decorator
    def _decorator(func):
        @wraps(func)
        def _wrapped(*args, **kwargs):
            try:
                monitor.record()
            except Exception:
                pass
            return func(*args, **kwargs)

        return _wrapped

    # If called directly, return the monitor instance
    if name is None:
        return monitor

    return _decorator


def get_performance_stats() -> Dict[str, Any]:
    return monitor.get_stats()


__all__ = ["PerformanceMonitor", "monitor_performance", "get_performance_stats", "EnhancedCache"]


class ParallelProcessor:
    """Minimal ParallelProcessor shim for dev/test.

    The real project provides a parallel processing utility; the shim
    accepts a callable and a list of inputs and invokes them sequentially
    (sufficient for smoke tests and import compatibility).
    """

    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max_workers

    def map(self, func, inputs):
        # Sequential fallback for shimbed processing.
        return [func(x) for x in inputs]


__all__.append("ParallelProcessor")


class OptimizedDatabaseQuery:
    """Shim for OptimizedDatabaseQuery compatibility.

    The real implementation may be async and database-backed. This shim
    exposes a small synchronous API suitable for import compatibility
    and smoke tests.
    """

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn

    def execute(self, query: str, params=None):
        return [{"query": query, "params": params, "status": "shimbed"}]


__all__.append("OptimizedDatabaseQuery")


class DummyEnhancedCache:
    """Very small in-memory cache shim used when the enhanced cache is unavailable.

    Exposes `get_stats()` to satisfy `api_server.cache_stats()` and a minimal
    `get`/`set` surface if needed by other code paths.
    """

    def __init__(self):
        self._store = {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value, ttl_seconds: int | None = None):
        self._store[key] = value

    def get_stats(self):
        return {"items": len(self._store), "metrics_available": True}


__all__.append("DummyEnhancedCache")

# If the imported EnhancedCache isn't available, provide a local dummy fallback
try:
    EnhancedCache  # type: ignore
except NameError:
    EnhancedCache = DummyEnhancedCache
