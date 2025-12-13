"""Stub for core.response_optimizer used by api_server."""


class ResponseOptimizer:
    def process(self, resp):
        return resp


def get_response_optimizer():
    return ResponseOptimizer()
