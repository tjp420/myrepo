"""Shim for `agi_chatbot.metrics.agi_readiness`.

Provides a minimal `compute_readiness` helper.
"""

from typing import Any, Dict


def compute_readiness(metrics: Dict[str, Any]) -> float:
    return 1.0
