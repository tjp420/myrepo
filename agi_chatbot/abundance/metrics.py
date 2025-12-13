"""Shim for `agi_chatbot.abundance.metrics`.

Provides minimal `snapshot` and `compute` helpers.
"""

from typing import Any, Dict


def snapshot() -> Dict[str, Any]:
    return {}


def compute(data: Any) -> Dict[str, Any]:
    return {"computed": True}
