"""Stub for response_generator used by api_server."""


def generate_response(prompt: str):
    return ""


class ResponseGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt: str):
        return generate_response(prompt)
