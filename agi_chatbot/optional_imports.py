"""Small helper for optional imports with deduplicated warnings.

Usage:
    from .optional_imports import optional_import

    # Get an attribute from an optional module (falls back to ``None``):
    optimize_fn = optional_import(
        'divine_response_optimizer',
        attr='optimize_ai_response',
        fallback=None,
        warn_msg='Divine optimizer not available',
    )

This module avoids noisy repeated prints by remembering which warnings
have already been emitted for the process lifetime.
"""

import importlib
import logging
from typing import Any, Optional

_WARNED = set()


def optional_import(
    module_name: str,
    attr: Optional[str] = None,
    fallback: Any = None,
    warn_msg: Optional[str] = None,
) -> Any:
    """Attempt to import ``module_name`` (or ``module_name.attr``).

    - If the import succeeds, returns the module or attribute.
    - On ``ImportError`` (or ``AttributeError`` when ``attr`` is provided),
      returns ``fallback``.
    - If ``warn_msg`` is provided, logs it only once per process lifetime.
    """
    logger = logging.getLogger(__name__)
    try:
        mod = importlib.import_module(module_name)
        if attr:
            try:
                return getattr(mod, attr)
            except AttributeError:
                if warn_msg and warn_msg not in _WARNED:
                    _WARNED.add(warn_msg)
                    try:
                        logger.warning(warn_msg)
                    except Exception:
                        print(warn_msg)
                return fallback
        return mod
    except ImportError:
        if warn_msg and warn_msg not in _WARNED:
            _WARNED.add(warn_msg)
            try:
                logger.warning(warn_msg)
            except Exception:
                print(warn_msg)
        return fallback
