"""Dev shim: explainability engine placeholder.

Exports `get_explainability_engine` for dev/testing.
"""

from typing import Any, Dict


class _ExplainabilityEngine:
    def explain(self, *args, **kwargs) -> Dict[str, Any]:
        return {"explanation": "dev-shim"}

    def start_reasoning_trace(self, *args, **kwargs) -> None:
        return None

    def add_reasoning_step(self, *args, **kwargs) -> None:
        return None

    def complete_reasoning_trace(self, *args, **kwargs) -> None:
        return None

    def get_reasoning_explanation(self, *args, **kwargs) -> Dict[str, Any]:
        return {"explanation": []}

    def perform_transparency_audit(self, *args, **kwargs) -> Dict[str, Any]:
        return {"audit": {}}

    def get_transparency_dashboard_data(self, *args, **kwargs) -> Dict[str, Any]:
        return {"panels": []}


def get_explainability_engine() -> Any:
    return _ExplainabilityEngine()


def explain_decision(decision):
    return {}


"""Dev shim: explainability engine placeholder.

Exports `get_explainability_engine` for dev/testing.
"""
from typing import Any, Dict


class _ExplainabilityEngine:
    def explain(self, *args, **kwargs):
        return {"explanation": "dev-shim"}

    def start_reasoning_trace(self, *args, **kwargs) -> None:
        return None

    def add_reasoning_step(self, *args, **kwargs) -> None:
        return None

    def complete_reasoning_trace(self, *args, **kwargs) -> None:
        return None

    def get_reasoning_explanation(self, *args, **kwargs) -> Dict[str, Any]:
        return {"explanation": []}

    def perform_transparency_audit(self, *args, **kwargs) -> Dict[str, Any]:
        return {"audit": {}}

    def get_transparency_dashboard_data(self, *args, **kwargs) -> Dict[str, Any]:
        return {"panels": []}


def get_explainability_engine() -> Any:
    return _ExplainabilityEngine()


"""Stub for core.explainability used by api_server."""


def explain_decision(decision):
    return {}
