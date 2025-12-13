"""Stub for core.strategy_adaptation used in api_server imports."""


def adapt_strategy(*args, **kwargs):
    return None


class StrategyAdaptationEngine:
    def adapt(self, *args, **kwargs):
        return adapt_strategy(*args, **kwargs)
