"""Dev shim for configuration paths referenced across the codebase.
Provides `data_path` and related names so imports resolve in dev mode.
"""

from pathlib import Path

data_path = Path("./data")
"""Configuration paths shim used by api_server imports."""


def get_data_path():
    return None
