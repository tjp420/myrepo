"""Memory shim for reasoning subpackage."""


def store_step(step: str):
    return None


def get_memory_system():
    class _Memory:
        def get(self, k, default=None):
            return default

        def set(self, k, v):
            pass

    return _Memory()
