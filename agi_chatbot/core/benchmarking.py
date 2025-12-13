"""Minimal benchmarking registry shim for dev/test.

Provides a lightweight `registry` object so `api_server` can import
`micro_benchmark_registry` during TestClient runs.
"""

from typing import Any, Callable, Dict, List


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._benchmarks: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        self._benchmarks[name] = func

    def list(self) -> List[str]:
        return list(self._benchmarks.keys())

    def run(self, name: str, *args, **kwargs) -> Any:
        fn = self._benchmarks.get(name)
        if fn is None:
            raise KeyError(f"Benchmark '{name}' not found")
        return fn(*args, **kwargs)


# A simple, usable registry instance for imports that expect a registry object.
registry = BenchmarkRegistry()


__all__ = ["BenchmarkRegistry", "registry"]
