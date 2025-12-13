"""Latency utilities shim for development and testing.

Provides `sleep_jitter_from_env` used by the API server. The real
project implementation may inspect several environment variables and
introduce a short asyncio sleep to simulate latency. For development
and in-process tests we provide a safe, lightweight async shim that
accepts the same call signature used in `api_server.py`.
"""

import asyncio
import os
import random
from typing import Optional


async def sleep_jitter_from_env(
    env_var: str = "AGI_API_LATENCY_MS", min_ms: int = 0, max_ms: int = 0
) -> None:
    """Async no-op or small jittered sleep controlled by environment.

    Expected usages call this as: ``await sleep_jitter_from_env("AGI_API_LATENCY_MS", 0, 0)``
    - If `AGI_DEV` is set (truthy), this function returns immediately.
    - If the named env var is present and contains "min,max" (ms), that
      range overrides the provided ``min_ms``/``max_ms`` values.
    - If both min and max are zero (and no env override), this is a no-op.
    """
    # Fast-path for dev/test: skip sleeping when developer mode is enabled
    dev_flag = os.getenv("AGI_DEV")
    if dev_flag and dev_flag.lower() in ("1", "true", "yes"):
        return

    # Allow the environment var to override the min/max values.
    val: Optional[str] = os.getenv(env_var)
    if val:
        try:
            if "," in val:
                parts = [int(p.strip()) for p in val.split(",", 1)]
                if len(parts) == 2:
                    min_ms, max_ms = parts[0], parts[1]
                else:
                    min_ms = max_ms = parts[0]
            else:
                # single numeric value -> use as both min and max
                num = int(val.strip())
                min_ms = min_ms or num
                max_ms = max_ms or num
        except Exception:
            # On parse errors, fall back to provided arguments
            pass

    # Normalize and early-exit if there's nothing to do
    if max_ms < min_ms:
        max_ms = min_ms
    if max_ms == 0 and min_ms == 0:
        return

    # Choose jitter in milliseconds and sleep asynchronously
    jitter_ms = random.randint(min_ms, max_ms)
    await asyncio.sleep(jitter_ms / 1000.0)


__all__ = ["sleep_jitter_from_env"]
