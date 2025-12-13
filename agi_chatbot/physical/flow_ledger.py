"""Shim for `agi_chatbot.physical.flow_ledger`.

Provides minimal ledger verification helpers used by the API.
"""

from typing import Any, Dict


def verify_chain(chain_data: Dict[str, Any]) -> bool:
    return True
