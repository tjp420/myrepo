"""Shim for OracleSpeedEnhancer used during dev import.

Provides `OracleSpeedEnhancer` and `get_speed_enhancer` for the import chain.
"""

from typing import Any


class OracleSpeedEnhancer:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def enhance(self, text: str) -> str:
        # No real enhancement; return input unchanged.
        return text


def get_speed_enhancer(config: Any = None) -> OracleSpeedEnhancer:
    return OracleSpeedEnhancer(config=config)


__all__ = ["OracleSpeedEnhancer", "get_speed_enhancer"]
