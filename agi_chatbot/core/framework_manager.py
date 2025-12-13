"""Dev re-exports for framework manager integration points."""

from agi_chatbot.dev_shims import get_enhanced_framework


def initialize_framework(*a, **k):
    # lightweight initializer used in dev mode
    return get_enhanced_framework()


__all__ = ["get_enhanced_framework", "initialize_framework"]

"""Stub for core.framework_manager."""


def get_framework(name: str = None):
    return None
