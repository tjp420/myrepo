"""Minimal shim for agent benchmarks used by the API server.

This file provides a lightweight stub of the real benchmarking utilities
so the TestClient can import `agi_chatbot.api_server` during dev-mode.
"""

from typing import Any, Dict, List

DEFAULT_TASKS: List[Dict[str, Any]] = [
    {"id": "hello_world", "description": "Print Hello World", "difficulty": "easy"},
]


class CodingBenchmarkRunner:
    """A tiny shim runner that pretends to run coding benchmarks.

    The real project provides a full runner; for dev/test we only need
    a lightweight object with a compatible interface.
    """

    def __init__(self, tasks: List[Dict[str, Any]] = None) -> None:
        self.tasks = tasks or DEFAULT_TASKS

    def run(self) -> Dict[str, Any]:
        # Return a simple, deterministic result suitable for smoke tests.
        results = []
        for t in self.tasks:
            results.append(
                {"task_id": t.get("id"), "status": "skipped", "notes": "shimbed"}
            )
        return {
            "summary": {"total": len(self.tasks), "completed": 0},
            "results": results,
        }


__all__ = ["CodingBenchmarkRunner", "DEFAULT_TASKS"]
