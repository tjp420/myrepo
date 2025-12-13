"""Dev re-export for semantic engine used by api_server."""

from agi_chatbot.dev_shims import get_semantic_engine

__all__ = ["get_semantic_engine"]
"""Stub for knowledge.semantic_engine used in api_server."""


def semantic_search(query: str):
    return []
