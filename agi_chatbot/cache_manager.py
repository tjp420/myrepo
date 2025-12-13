"""Minimal cache_manager shim exposing EnhancedCache for api_server imports."""

import asyncio
import inspect
from functools import wraps
from typing import Any, Dict, Optional


class EnhancedCache:
    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get_stats(self) -> Dict[str, int]:
        return {"size": len(self._store)}


def get_enhanced_cache() -> EnhancedCache:
    return EnhancedCache()


__all__ = ["EnhancedCache", "get_enhanced_cache"]


def cache_response_decorator(ttl_seconds: int = 0):
    """Return a simple decorator that would cache responses for `ttl_seconds`.

    This shim does not implement caching semantics; it's a no-op decorator
    sufficient for import-time compatibility and smoke tests.
    """

    def _decorator(func):
        # Always return an async wrapper so FastAPI treats endpoints as async
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def _wrapped(*args, **kwargs):
                return await func(*args, **kwargs)

            return _wrapped

        @wraps(func)
        async def _wrapped_sync(*args, **kwargs):
            return await asyncio.to_thread(func, *args, **kwargs)

        return _wrapped_sync

    return _decorator


__all__.append("cache_response_decorator")


# Backwards-compatible name expected by older imports
def cache_response(*args, **kwargs):
    """Alias for `cache_response_decorator` to preserve older import names."""
    return cache_response_decorator(*args, **kwargs)


__all__.append("cache_response")
