"""Minimal async API client shim providing `OptimizedDatabaseQuery`.

This shim satisfies imports in `api_server` that expect an async-DB
query helper. The shim provides a synchronous-compatible placeholder
that returns deterministic results for smoke tests.
"""

from typing import Any, Dict, List, Optional


class OptimizedDatabaseQuery:
    """Placeholder for an optimized async DB query helper.

    Real implementation may provide async queries, pooling and retries.
    The shim provides a simple `run` method returning a static result.
    """

    def __init__(self, dsn: Optional[str] = None, max_workers: int = 1) -> None:
        self.dsn = dsn
        self.max_workers = max_workers

        # Provide a minimal query cache attribute used by callers in api_server
        class _QueryCache:
            def __init__(self):
                self._store = {}

            def _evict_if_needed(self):
                return None

            def clear(self):
                try:
                    self._store.clear()
                except Exception:
                    self._store = {}

            def get(self, k, default=None):
                return self._store.get(k, default)

            def set(self, k, v):
                self._store[k] = v

        self.query_cache = _QueryCache()

    def run(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Return a deterministic placeholder result for smoke tests.
        return [{"query": query, "params": params, "result": "shim"}]

    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Compatibility alias used by `api_server` and other modules.

        Delegates to `run` to provide the expected API surface for
        synchronous call-sites and static analysis.
        """
        try:
            return self.run(query, params)
        except Exception:
            return []


__all__ = ["OptimizedDatabaseQuery"]
