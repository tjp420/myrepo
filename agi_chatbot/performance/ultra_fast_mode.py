"""Lightweight shim for `agi_chatbot.performance.ultra_fast_mode`.

This file provides a callable `ultra_fast_enabled()` used by the
application code, plus a small helper to enable the mode during tests.
The real implementation in production exposes more complex behavior;
this shim keeps tests deterministic.
"""

_ULTRA_FAST_ENABLED = False

log_hint_prefix = "[ULTRA_FAST_SHIM]"

def ultra_fast_enabled() -> bool:
    """Return whether ultra-fast mode is enabled.

    Kept as a callable to match the production API used by
    `api_server.py` (the code expects to call this as a function).
    """
    return bool(_ULTRA_FAST_ENABLED)

def enable_ultra_fast() -> None:
    """Enable ultra-fast mode for tests or runtime configuration."""
    global _ULTRA_FAST_ENABLED
    _ULTRA_FAST_ENABLED = True

def disable_ultra_fast() -> None:
    """Disable ultra-fast mode."""
    global _ULTRA_FAST_ENABLED
    _ULTRA_FAST_ENABLED = False
