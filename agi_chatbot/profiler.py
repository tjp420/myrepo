"""Dev re-export for profiler APIs referenced in api_server."""

from agi_chatbot.dev_shims import DummyProfiler


def get_profiler() -> DummyProfiler:
    return DummyProfiler()


__all__ = ["get_profiler", "DummyProfiler"]
