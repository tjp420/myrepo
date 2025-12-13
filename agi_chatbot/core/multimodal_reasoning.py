"""Stub for core.multimodal_reasoning used by api_server."""


def reason_multimodal(*args, **kwargs):
    return None


class MultimodalReasoning:
    def __init__(self, *args, **kwargs):
        pass

    def reason(self, *args, **kwargs):
        return reason_multimodal(*args, **kwargs)

    def analyze(self, *args, **kwargs):
        return self.reason(*args, **kwargs)
