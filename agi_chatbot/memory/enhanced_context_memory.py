"""Minimal enhanced_context_memory shim for tests and linting."""


class EnhancedContextMemory:
    def __init__(self, *args, **kwargs):
        self._store = {}

    def get(self, key, default=None):
        return self._store.get(key, default)

    def set(self, key, value):
        self._store[key] = value


def get_enhanced_context_memory():
    return EnhancedContextMemory()


# Backwards-compatible manager class expected by imports
class EnhancedContextMemoryManager(EnhancedContextMemory):
    def touch(self, key):
        return None
