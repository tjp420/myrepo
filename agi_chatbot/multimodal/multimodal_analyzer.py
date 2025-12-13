"""Stub for multimodal.multimodal_analyzer used by api_server."""


def analyze_multimodal(data):
    return {}


class MultiModalAnalyzer:
    def __init__(self, *args, **kwargs):
        pass

    def analyze(self, data):
        return analyze_multimodal(data)

    def get_capabilities(self):
        return {"multimodal": False}

    def get_installation_guide(self):
        # Return a mapping of optional dependencies -> installation hints.
        # Production implementation can return richer details; keep shim simple.
        return {}
