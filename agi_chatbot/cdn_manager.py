"""Minimal CDN manager shim for dev/test.

Provides a `CDNManager` class expected by `api_server`.
"""

from typing import Any, Dict


class CDNManager:
    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def upload(self, path: str, content: bytes) -> str:
        # Return a fake CDN URL for the uploaded content.
        return f"https://cdn.example.local/{path}"

    def delete(self, url: str) -> bool:
        return True


__all__ = ["CDNManager"]
