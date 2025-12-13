"""Dev shim exports for enhanced semantic cache API."""

from agi_chatbot.dev_shims import EnhancedCache, get_enhanced_cache

__all__ = ["get_enhanced_cache", "EnhancedCache"]
"""Stub for enhanced_semantic_cache used by api_server."""


class EnhancedSemanticCache:
    def __init__(self):
        self._store = {}

    def get(self, k, default=None):
        return self._store.get(k, default)

    def set(self, k, v):
        self._store[k] = v


def get_enhanced_semantic_cache():
    return EnhancedSemanticCache()
