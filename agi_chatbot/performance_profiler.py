import time
from contextlib import contextmanager
from functools import wraps


class DummyProfiler:
    def start(self):
        return None

    def stop(self):
        return None

    @contextmanager
    def profile(self, name=None):
        yield


def get_profiler():
    """Return a dummy profiler compatible with expected interface."""
    return DummyProfiler()


def profile_endpoint(func=None, *, name=None):
    """Decorator used to profile endpoint handlers. No-op in shim."""
    if func is None:

        def wrapper(f):
            @wraps(f)
            def inner(*a, **kw):
                return f(*a, **kw)

            return inner

        return wrapper

    @wraps(func)
    def inner(*a, **kw):
        return func(*a, **kw)

    return inner


def profile_cache(func=None):
    return profile_endpoint(func)


def profile_db(func=None):
    return profile_endpoint(func)


def timeit(name=None):
    """Simple timing context manager for local use."""

    @contextmanager
    def _cm():
        t0 = time.time()
        try:
            yield
        finally:
            _ = time.time() - t0

    return _cm()
