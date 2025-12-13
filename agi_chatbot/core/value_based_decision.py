"""Stub for core.value_based_decision."""


def decide_value_based(*args, **kwargs):
    return None


class _ValueBasedDecisionEngine:
    def decide(self, *args, **kwargs):
        return decide_value_based(*args, **kwargs)


def get_value_based_decision_engine():
    return _ValueBasedDecisionEngine()
