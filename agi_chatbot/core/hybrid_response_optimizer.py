"""Dev shim: hybrid response optimizer minimal API.

This file exposes `optimize_ai_response` used by the server in several
places. The implementation is intentionally tiny and safe for dev.
"""

from typing import Any


def optimize_ai_response(response: Any, *args, **kwargs) -> Any:
    """Return the input response unchanged (dev/no-op)."""
    return response


"""Stub for core.hybrid_response_optimizer used by api_server."""


class HybridResponseOptimizer:
    def optimize(self, response):
        return response


def get_hybrid_optimizer():
    return HybridResponseOptimizer()
