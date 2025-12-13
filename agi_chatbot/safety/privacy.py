"""Shim for `agi_chatbot.safety.privacy`.

Provides minimal redaction helpers used by imports.
"""

from typing import Any


def redact_text(text: str) -> str:
    # Minimal redaction: return the original text for tests
    return text


def redact_obj(obj: Any) -> Any:
    # Return object unchanged in shim
    return obj
