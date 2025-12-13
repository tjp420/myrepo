"""Shim for `agi_chatbot.safety.formal_verification`.

Provides minimal formal verification helpers used by imports.
"""

from typing import Any, Dict


def verify_safety_properties(model_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {"verified": True, "issues": []}


def verify_property(model_spec: Dict[str, Any], prop: str) -> bool:
    return True
