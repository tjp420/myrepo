"""Shim for `agi_chatbot.core.echoes_evaluation` used by the app imports.

Provides minimal evaluation helpers for echoes sessions.
"""

from typing import Any, Dict


def evaluate_echoes_session(session_id: str) -> Dict[str, Any]:
    return {"session_id": session_id, "score": 0.0, "notes": "shim"}


def get_echoes_platform_analytics(session_id: str) -> Dict[str, Any]:
    return {"session_id": session_id, "analytics": {}}


def generate_echoes_evaluation_report(session_id: str) -> str:
    return "echoes-eval-report-shim"
