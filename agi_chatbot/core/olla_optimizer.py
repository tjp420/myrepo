"""Shim for OLLAOptimizer used by the API server.

This is a minimal placeholder that exposes the class name expected
by the import chain so dev-mode imports succeed.
"""

from typing import Any


class OLLAOptimizer:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def optimize(self, prompt: str) -> str:
        # No real optimization in the shim; return the input unchanged.
        return prompt


__all__ = ["OLLAOptimizer"]
