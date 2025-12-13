"""Minimal error_handler shim used during in-process testing.

Provides a no-op registration function so imports succeed.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI):
    """Register basic exception handlers. No-op for shim."""
    # In production this would add exception handlers/middlewares.
    logger.debug("register_error_handlers called (shim)")


def handle_exception(request: Request, exc: Exception):
    logger.exception("Unhandled exception (shim)")
    return {"detail": "internal server error"}


class ErrorHandler:
    """Minimal ErrorHandler shim providing expected interface."""

    def __init__(self, app: FastAPI | None = None):
        self.app = app

    def register(self, app: FastAPI):
        self.app = app
        # No-op for shim: real implementation would add handlers/middlewares
        logger.debug("ErrorHandler.register called (shim)")

    def handle(self, request: Request, exc: Exception):
        return handle_exception(request, exc)


# Backwards-compatible alias used in some places
class RetryingErrorHandler(ErrorHandler):
    """Shim for a retriable/robust error handler class expected by code elsewhere."""

    def __init__(self, app: FastAPI | None = None, retries: int = 3):
        super().__init__(app)
        self.retries = retries

    def register(self, app: FastAPI):
        super().register(app)
        logger.debug("RetryingErrorHandler.register called (shim)")


def create_exception_handlers(app: FastAPI, handler=None):
    """Return a mapping of exception handlers. Shim returns simple handler mapping.

    Accept an optional `handler` argument to match the real function
    signature used by `api_server`.
    """

    def _validation_exc_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500, content={"detail": "validation error (shim)"}
        )

    return {
        Exception: lambda request, exc: JSONResponse(
            status_code=500, content={"detail": "internal error (shim)"}
        ),
        RequestValidationError: _validation_exc_handler,
    }


def create_logging_middleware(app: FastAPI):
    """Create a basic logging middleware placeholder. No-op for shim."""
    logger.debug("create_logging_middleware called (shim)")
    return None


class AGIChatbotError(Exception):
    """Base exception type used by AGI chatbot error handling."""


class AGIError(AGIChatbotError):
    """Alias for AGIChatbotError for backwards compatibility."""


def __getattr__(name: str):
    """Dynamically provide fallback symbols expected by imports.

    - If the name ends with 'Error' return a simple Exception subclass.
    - Otherwise return a no-op callable.
    This keeps imports working during in-process testing without adding many
    dedicated shim classes.
    """
    if name.endswith("Error"):
        return type(name, (AGIChatbotError,), {})

    def _noop(*args, **kwargs):
        logger.debug(f"Called shimbed symbol {name}")
        return None

    return _noop
