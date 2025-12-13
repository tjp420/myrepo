"""Stub for context.context_analyzer used by api_server."""


def analyze_context(context):
    return {}


class ContextAnalyzer:
    def __init__(self, *args, **kwargs):
        pass

    def analyze(self, context):
        return analyze_context(context)
