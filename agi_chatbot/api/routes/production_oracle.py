"""Production oracle route shim used by api_server imports."""

from fastapi import APIRouter

# Tests and static analysis expect a `router` symbol; expose a minimal
# APIRouter instance to satisfy imports.
router = APIRouter()


def register_routes(app):
    # no-op registration for tests
    return None
