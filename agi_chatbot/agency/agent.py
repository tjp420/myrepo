"""Minimal shim for `agi_chatbot.agency.agent` used during TestClient imports.

Exports `run_task` and `AgentConfig` to satisfy imports.
"""

from dataclasses import dataclass
from typing import Any


def run_task(*args, **kwargs) -> Any:
    """No-op run_task shim used by tests."""
    return None


@dataclass
class AgentConfig:
    name: str = "agent-shim"
    max_retries: int = 0
