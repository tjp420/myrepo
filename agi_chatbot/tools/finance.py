"""Shim for `agi_chatbot.tools.finance` used by api_server imports.

Provides a minimal `get_indicators` function.
"""

from typing import Any, Dict


def get_indicators(symbol: str) -> Dict[str, Any]:
    return {"symbol": symbol, "indicators": {}}
