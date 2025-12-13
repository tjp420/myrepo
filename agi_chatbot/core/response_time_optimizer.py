"""Shim for response_time_optimizer used by api_server imports.

This provides a minimal `get_response_optimizer` factory returning
an object with a no-op `optimize` method and a decorator helper.
"""

from functools import wraps


class DummyOptimizer:
    def optimize(self, *a, **kw):
        return None

    def wrap(self, func):
        @wraps(func)
        def _inner(*args, **kwargs):
            return func(*args, **kwargs)

        return _inner


def get_response_optimizer():
    return DummyOptimizer()
