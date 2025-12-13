"""Dev re-exports for oracle/performance integration points."""

from agi_chatbot.dev_shims import get_adaptive_optimizer

__all__ = ["get_adaptive_optimizer"]
"""Stub for core.oracle_performance_wisdom used by api_server."""


def get_wisdom_stats(*args, **kwargs):
    return {}
