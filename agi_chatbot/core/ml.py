"""Minimal ml shim used by api_server imports.

Expose a small `ResponsePredictor` class used by `api_server`.
"""


def predict(*args, **kwargs):
    return None


class ResponsePredictor:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        return None
