"""Shim for conversational AI improvement utilities used during import.

Provides minimal `OracleInspiredOptimizer` and `ConversationContext`.
"""

from typing import Any, Dict


class OracleInspiredOptimizer:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def improve(self, message: str) -> str:
        return message


class ConversationContext:
    def __init__(self, user_id: str = "dev") -> None:
        self.user_id = user_id
        self.history: Dict[str, str] = {}

    def add(self, role: str, text: str) -> None:
        self.history[role] = text


__all__ = ["OracleInspiredOptimizer", "ConversationContext"]
