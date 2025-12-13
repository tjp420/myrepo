"""Helper utilities for graceful handling of optional imports.

Provide a single place to attempt optional imports and emit a single
informative warning when a dependency is missing.
"""

import importlib
import logging
from typing import Optional

_warned = set()


def optional_import(
    module_name: str, alias: Optional[str] = None, warn_msg: Optional[str] = None
):
    """Try to import `module_name` and return the module or None.

    If the import fails, log a single warning message (per `module_name`).
    """
    logger = logging.getLogger(__name__)
    try:
        mod = importlib.import_module(module_name)
        if alias:
            # also place in globals? caller can assign as needed
            pass
        return mod
    except Exception:
        key = module_name
        if key not in _warned:
            _warned.add(key)
            msg = (
                warn_msg
                or f"Optional dependency '{module_name}' is not available; some features will be disabled."
            )
            logger.warning(msg)
        return None
