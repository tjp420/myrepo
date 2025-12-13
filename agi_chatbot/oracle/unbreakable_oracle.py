"""Stub for agi_chatbot.oracle.unbreakable_oracle.

This file provides a permissive, import-time-safe UnbreakableOracle
implementation used for linting and lightweight tests.
"""

from types import SimpleNamespace


def answer_unbreakable(query: str):
    return None


class UnbreakableOracle:
    def __init__(self, *args, **kwargs):
        pass

    def ask(self, query: str):
        return answer_unbreakable(query)

    def validate_capability(self, capability: str) -> bool:
        return False

    def generate_reasoning_trace(self, *args, **kwargs):
        return []

    def audit_response(self, *args, **kwargs):
        return {"safe": True}

    def query(self, *args, **kwargs):
        return None

    def get_transparency_report(self, *args, **kwargs):
        return {}

    def get_reasoning_explanation(self, *args, **kwargs):
        return {"explanation": []}

    def process_input(self, *args, **kwargs):
        # Return a lightweight response-like object expected by callers
        return SimpleNamespace(
            response=None,
            confidence=0.0,
            processing_time=0.0,
            source="stub",
            metadata={},
        )

    async def process_input_async(self, *args, **kwargs):
        return self.process_input(*args, **kwargs)

    def add_knowledge(self, *args, **kwargs):
        return None

    def get_stats(self, *args, **kwargs):
        return {"queries": 0}

    def is_initialized(self, *args, **kwargs):
        return False

    def optimize_performance(self, *args, **kwargs):
        return {"optimized": 0, "recommendations": []}

    def respond(self, *args, **kwargs):
        return {"response": None}

    def _assess_query_risk(self, *args, **kwargs):
        return {"risk": "low"}

    def get_cached_response(self, *args, **kwargs):
        return None

    def cache_response(self, *args, **kwargs):
        return None

    def update_weights(self, *args, **kwargs):
        return None

    @property
    def knowledge_base(self):
        return SimpleNamespace(entries=[])

    @property
    def nlp_engine(self):
        return SimpleNamespace(is_initialized=False)

    @property
    def performance_monitor(self):
        return SimpleNamespace(get_metrics=lambda *a, **k: {})
