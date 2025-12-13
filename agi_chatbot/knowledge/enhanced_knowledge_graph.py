"""Re-export shim for knowledge graph APIs used in dev mode."""

from agi_chatbot.dev_shims import get_knowledge_graph

__all__ = ["get_knowledge_graph"]
"""Stub for knowledge.enhanced_knowledge_graph used by api_server."""


def build_enhanced_graph(*args, **kwargs):
    return {}
