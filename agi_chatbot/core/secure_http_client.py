"""Minimal secure HTTP client shim for import-time compatibility.

Exports small config dataclasses and factory functions used by the API server.
"""

from typing import Any, Dict


class HttpClientConfig:
    def __init__(self, base_url: str = "http://localhost", timeout: int = 5) -> None:
        self.base_url = base_url
        self.timeout = timeout


class AuthConfig:
    def __init__(self, token: str = "") -> None:
        self.token = token


class SecureHttpClient:
    def __init__(
        self, config: HttpClientConfig = None, auth: AuthConfig = None
    ) -> None:
        self.config = config or HttpClientConfig()
        self.auth = auth or AuthConfig()

    def get(self, path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        # Return a simple placeholder response.
        return {"status": "ok", "path": path, "params": params}


def get_secure_client(
    config: HttpClientConfig = None, auth: AuthConfig = None
) -> SecureHttpClient:
    return SecureHttpClient(config=config, auth=auth)


def get_http_client_snippets() -> Dict[str, str]:
    return {"example": "client.get('/health')"}


__all__ = [
    "HttpClientConfig",
    "AuthConfig",
    "SecureHttpClient",
    "get_secure_client",
    "get_http_client_snippets",
]
