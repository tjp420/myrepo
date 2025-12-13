"""Stub for core.oracle_reality_processor.

Provide a lightweight `OracleRealityProcessor` implementation used by
the API surface during tests and lint-run passes.
"""


class OracleRealityProcessor:
    def __init__(self, *args, **kwargs):
        pass

    def analyze(self, payload):
        return {"status": "unavailable", "payload": payload}


def process_reality(data):
    return data
