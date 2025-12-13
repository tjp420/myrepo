"""Shim for parallel response engine utilities used during import.

Provides `parallel_process_requests` used by the API server.
"""

from typing import Any, List


def parallel_process_requests(requests: List[Any]) -> List[Any]:
    # Simple sequential processing for the shim: return inputs unchanged.
    return list(requests)


__all__ = ["parallel_process_requests"]
