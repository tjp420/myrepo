"""Shim for FAISS ANN index utilities used by the API server.

Provides `get_global_ann_index` and `_FAISS_AVAILABLE` to satisfy imports
during dev-mode TestClient runs.
"""

from typing import Any, Tuple

_FAISS_AVAILABLE = False


def get_global_ann_index() -> Tuple[bool, Any]:
    """Return a tuple (available, index) where `index` is a shim object.

    The real function returns an actual FAISS index; for tests we return
    (False, None) to indicate FAISS is not available.
    """
    return _FAISS_AVAILABLE, None


__all__ = ["get_global_ann_index", "_FAISS_AVAILABLE"]
