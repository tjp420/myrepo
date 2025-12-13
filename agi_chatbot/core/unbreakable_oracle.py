"""Dev shim: unbreakable oracle helper.

Provides a minimal `get_unbreakable_oracle` factory used by the API
in dev mode so pylint and runtime import resolution succeed.
"""

from typing import Any

try:
    from agi_chatbot.dev_shims import UnbreakableOracle  # type: ignore
except Exception:

    class UnbreakableOracle:  # fallback local definition
        def __init__(self):
            pass

        def ask(self, *a, **k):
            return None


def get_unbreakable_oracle() -> Any:
    """Return a lightweight UnbreakableOracle instance for dev-mode."""
    try:
        return UnbreakableOracle()
    except Exception:
        return UnbreakableOracle()


"""Stub for core.unbreakable_oracle."""


def consult_unbreakable_oracle(query: str):
    return None
