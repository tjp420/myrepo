"""Minimal Oracle API shim for imports.

Provides a lightweight `Oracle` class used by api_server.
"""


class Oracle:
    def __init__(self, *args, **kwargs) -> None:
        # Accept flexible constructor signatures used by the API server.
        self.config = (
            kwargs.get("config") if "config" in kwargs else (args[0] if args else None)
        )

    def answer(self, prompt: str) -> str:
        return "shimbed-answer"


def oracle(prompt: str) -> str:
    return Oracle().answer(prompt)


__all__ = ["Oracle", "oracle"]
