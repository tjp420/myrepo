"""Stub for core.temporal_conversation_state."""


def get_state(session_id: str):
    return {}


class TemporalConversationState:
    def __init__(self, session_id: str = None):
        self.session_id = session_id

    def to_dict(self):
        return {"session_id": self.session_id}
