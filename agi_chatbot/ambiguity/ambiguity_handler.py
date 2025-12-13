"""Stub for ambiguity.ambiguity_handler used by api_server."""


def resolve_ambiguity(text: str):
    return text


class AmbiguityHandler:
    def __init__(self, *args, **kwargs):
        pass

    def handle(self, text: str):
        return resolve_ambiguity(text)
