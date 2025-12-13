"""Shim for advanced response optimizer used during imports.

Provides `AdvancedResponseOptimizer` and `get_response_optimizer` used by the
API server in dev-mode smoke tests.
"""

from typing import Any


class AdvancedResponseOptimizer:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def optimize(self, response: str) -> str:
        # No-op optimization for the shim.
        return response


def get_response_optimizer(config: Any = None) -> AdvancedResponseOptimizer:
    return AdvancedResponseOptimizer(config=config)


__all__ = ["AdvancedResponseOptimizer", "get_response_optimizer"]
