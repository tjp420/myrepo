"""Shim for `agi_chatbot.tools.resources`.

Provides a minimal `recommend_resources` function used by the app.
"""

from typing import Any, Dict, List


def recommend_resources(topic: str) -> List[Dict[str, Any]]:
    return []
