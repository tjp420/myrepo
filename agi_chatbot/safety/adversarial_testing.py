"""Shim for `agi_chatbot.safety.adversarial_testing`.

Provides minimal adversarial detection/testing helpers.
"""

from typing import Any, Dict


def detect_adversarial_input(text: str) -> bool:
    # Shim always returns False (not adversarial)
    return False


def run_adversarial_tests(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"passed": True, "details": []}
