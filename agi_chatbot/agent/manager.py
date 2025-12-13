"""Shim for `agi_chatbot.agent.manager`.

Provides a minimal `get_agent_file_manager` helper used by the API.
"""

from typing import Any


def get_agent_file_manager(agent_id: str) -> Any:
    return None
