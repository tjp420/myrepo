"""Minimal shim for `agi_chatbot.agency.tools`.

Provides a `list_tools` function used by api_server imports.
"""

from typing import Dict, List


def list_tools() -> List[Dict[str, str]]:
    """Return a minimal empty tools list for testing."""
    return []
