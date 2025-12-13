"""Shim for `agi_chatbot.metrics.calibration`.

Provides minimal calibration utilities used by the API.
"""

from typing import Any, Dict


def compute_brier(predictions: Any, targets: Any) -> float:
    return 0.0


def record(entry: Dict[str, Any]) -> None:
    return None
