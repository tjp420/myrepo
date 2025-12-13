"""Middleware shim to sanitize Content-Encoding headers.

Provides a `ContentEncodingSanitizer` class compatible with
`app.add_middleware(...)` in FastAPI. The shim is minimal and safe for
dev/TestClient use.
"""

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class ContentEncodingSanitizer(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Call the next handler and sanitize response headers if needed.
        response: Response = await call_next(request)
        try:
            # If header exists and is malformed, remove it for safety.
            ce = response.headers.get("content-encoding")
            if ce and not isinstance(ce, str):
                # coerce to string or remove
                response.headers.pop("content-encoding", None)
        except Exception:
            logger.debug("content-encoding sanitization skipped (shim)")
        return response


__all__ = ["ContentEncodingSanitizer"]
