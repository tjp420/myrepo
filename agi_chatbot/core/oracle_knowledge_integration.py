"""Dev re-exports for oracle/knowledge integration points."""

from agi_chatbot.dev_shims import get_oracle_integration

__all__ = ["get_oracle_integration"]
"""Stub for core.oracle_knowledge_integration."""


def integrate_oracle_knowledge(*args, **kwargs):
    return None
