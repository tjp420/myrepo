"""Shim for performance monitoring utilities used by the API server.

Provides `PerformanceMonitor`, `monitor_performance`, and `get_performance_stats`.
"""

import inspect
from functools import wraps
from typing import Any, Dict


class PerformanceMonitor:
    def __init__(self) -> None:
        self._stats: Dict[str, Any] = {"requests": 0}

    def record_request(self) -> None:
        self._stats["requests"] += 1

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)


monitor = PerformanceMonitor()


def monitor_performance(name=None):
    """Decorator factory and accessor for the performance monitor (shim).

    Supports `@monitor_performance('tag')`, `@monitor_performance` and
    `monitor_performance()` usage.
    """

    if callable(name):
        func = name
        # Preserve async/sync nature of the wrapped function and
        # use functools.wraps so FastAPI can inspect the original
        # function signature instead of the wrapper's.
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def _wrapper(*args, **kwargs):
                try:
                    monitor.record_request()
                except Exception:
                    pass
                return await func(*args, **kwargs)

            return _wrapper

        @wraps(func)
        async def _wrapper(*args, **kwargs):
            try:
                monitor.record_request()
            except Exception:
                pass
            return await asyncio.to_thread(func, *args, **kwargs)

        return _wrapper

    def _decorator(func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def _wrapped(*args, **kwargs):
                try:
                    monitor.record_request()
                except Exception:
                    pass
                return await func(*args, **kwargs)

            return _wrapped

        @wraps(func)
        async def _wrapped(*args, **kwargs):
            try:
                monitor.record_request()
            except Exception:
                pass
            return await asyncio.to_thread(func, *args, **kwargs)

        return _wrapped

    if name is None:
        return monitor

    return _decorator


def get_performance_stats() -> Dict[str, Any]:
    return monitor.get_stats()


__all__ = ["PerformanceMonitor", "monitor_performance", "get_performance_stats"]
