"""Shim for `agi_chatbot.core.echoes_platform` used by the app imports.

Provides minimal no-op implementations of session and collaboration helpers.
"""

from typing import Any, Dict, List


def create_echoes_session(*args, **kwargs) -> Dict[str, Any]:
    return {"session_id": "shim-session", "status": "created"}


def collaborate_in_session(session_id: str, payload: Any) -> Any:
    return None


def submit_echoes_feedback(session_id: str, feedback: Dict[str, Any]) -> bool:
    return True


def get_echoes_session_summary(session_id: str) -> Dict[str, Any]:
    return {"session_id": session_id, "summary": "shim"}


def join_echoes_session(session_id: str, user_id: str) -> bool:
    return True


def leave_echoes_session(session_id: str, user_id: str) -> bool:
    return True


def get_echoes_session_participants(session_id: str) -> List[str]:
    return []


def get_echoes_ai_status(session_id: str) -> Dict[str, Any]:
    return {"ai": "idle"}


def get_echoes_security_status(session_id: str) -> Dict[str, Any]:
    return {"secure": True}


def get_echoes_audit_trail(session_id: str) -> List[Dict[str, Any]]:
    return []
