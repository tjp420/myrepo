"""Shim for `agi_chatbot.metrics.ledger`.

Provides minimal ledger helpers used by the API.
"""

from typing import Any, Dict


def get_ledger() -> Dict[str, Any]:
    return {}


def rollup_day(date_str: str) -> Dict[str, Any]:
    return {"date": date_str, "rolled": True}
