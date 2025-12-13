"""Shim for `agi_chatbot.scalability.distributed_cache`.

Provides minimal cache helper functions used by the API.
"""

from typing import Any


def get_cached_response(key: str) -> Any:
    return None


def cache_response(key: str, value: Any) -> None:
    return None
