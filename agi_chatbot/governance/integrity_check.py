"""Shim for `agi_chatbot.governance.integrity_check`.

Provides minimal governance helper functions used by the API.
"""

from typing import Any, Dict


def summarize(context: Dict[str, Any]) -> Dict[str, Any]:
    return {"summary": "shim"}


def should_enter_safe_mode(context: Dict[str, Any]) -> bool:
    return False
