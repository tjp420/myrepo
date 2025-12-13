"""Stub for agi_chatbot.oracle_mode to satisfy imports.

Expose a minimal `get_oracle_mode()` and `is_oracle_available()` for
consumers in `api_server`.
"""


def is_oracle_available() -> bool:
    return False


def get_oracle_mode() -> str:
    """Return a simple mode identifier for tests/dev."""
    return "disabled"
