"""Shim for UnbreakableOracleCodeOptimizer used during import.

Provides a minimal `UnbreakableOracleCodeOptimizer` for dev-mode imports.
"""

from typing import Any, Dict, Optional


class UnbreakableOracleCodeOptimizer:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def optimize_code(self, code: str) -> str:
        # No-op in shim.
        return code

    def calculate_sum_optimized(self, *args, **kwargs) -> Optional[int]:
        return None

    def minimize_memory_allocation(self, *args, **kwargs) -> None:
        return None

    def optimize_loops(self, *args, **kwargs) -> None:
        return None

    def get_optimization_stats(self, *args, **kwargs) -> Dict[str, Any]:
        return {"optimized_calls": 0}

    def reset_stats(self, *args, **kwargs) -> None:
        return None

    def clear_cache(self, *args, **kwargs) -> None:
        return None

    def parallelize_computations(self, *args, **kwargs) -> None:
        return None

    def vectorize_operations(self, *args, **kwargs) -> None:
        return None

    def optimize_algorithms(self, *args, **kwargs) -> None:
        return None

    def lazy_evaluation(self, *args, **kwargs) -> None:
        return None


__all__ = ["UnbreakableOracleCodeOptimizer"]
