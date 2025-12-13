"""Minimal core.chatbot shim to satisfy imports during linting/tests.

This file provides a lightweight, explicit `AGIChatbot` class so static
analysis (pylint) can reason about common attributes like `user_id` and
`user_name` referenced across the codebase.
"""

from typing import Any, Dict, Optional


class Chatbot:
    def __init__(self, *args, **kwargs):
        # Provide a minimal memory store used by the API surface
        self.memory: Dict[str, Any] = {}
        # Common profile attributes used by API code
        self.user_id: Optional[str] = None
        self.user_name: Optional[str] = None

    def respond(self, *args, **kwargs):
        return None


"""Minimal core.chatbot shim to satisfy imports during linting/tests.

This file provides a lightweight, explicit `AGIChatbot` class so static
analysis (pylint) can reason about common attributes like `user_id` and
`user_name` referenced across the codebase.
"""


class Chatbot:
    def __init__(self, *args, **kwargs):
        # Provide a minimal memory store used by the API surface
        self.memory: Dict[str, Any] = {}
        # Common profile attributes used by API code
        self.user_id: Optional[str] = None
        self.user_name: Optional[str] = None
        # Feature flags used by routing / caching logic
        self._use_semantic_cache: bool = False

    def respond(self, *args, **kwargs):
        return None


def get_chatbot():
    return Chatbot()


# Backwards-compatible name expected by imports
class AGIChatbot(Chatbot):
    def get_user_profile_info(self, user_id: Optional[str] = None):
        if user_id is None:
            # if no user_id provided, try instance attribute
            user_id = getattr(self, "user_id", None)
        return self.memory.get(user_id, {})

    def _ensure_user_profile(self):
        uid = getattr(self, "user_id", None)
        if uid is None:
            return {}
        self.memory.setdefault(uid, {})
        return self.memory[uid]

    def analyze_error(self, *args, **kwargs):
        return {}
