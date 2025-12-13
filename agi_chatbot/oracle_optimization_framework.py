"""Shim for the Unbreakable Oracle optimization framework.

Provides a minimal `UnbreakableOracleOptimizationFramework` class so
the api_server import chain can proceed in dev-mode.
"""

from typing import Any, Dict, Optional


class UnbreakableOracleOptimizationFramework:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def optimize(self, query: str) -> str:
        # No-op optimization for the shim.
        return query

    def optimize_query(self, *args, **kwargs) -> Optional[str]:
        return None

    def get_cached_response(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        return None

    def cache_response(self, *args, **kwargs) -> None:
        return None

    async def process_input_async(self, *args, **kwargs):
        # Return an empty list by default to satisfy call-sites that iterate
        return []

    def process_input(self, *args, **kwargs):
        return None

    def respond(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        return None

    def get_optimization_stats(self) -> Dict[str, Any]:
        return {"optimized": 0}

    def update_weights(self, *args, **kwargs) -> None:
        return None


__all__ = ["UnbreakableOracleOptimizationFramework"]
