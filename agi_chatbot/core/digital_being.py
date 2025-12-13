"""Minimal shim for core.digital_being used in imports.

Provides a small `DigitalBeing` class with a couple of safe methods.
"""

from typing import Any


class DigitalBeing:
    def __init__(self, *args, **kwargs):
        pass

    def get_status(self) -> dict:
        return {"status": "digital_being_stub", "available": False}

    def perform_action(self, *args, **kwargs) -> Any:
        return None
