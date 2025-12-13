"""Shim for OracleCodeEditor used during import.

Provides `OracleCodeEditor` and `get_oracle_editor` for dev-mode imports.
"""

from typing import Any


class OracleCodeEditor:
    def __init__(self, config: Any = None) -> None:
        self.config = config

    def edit(self, code: str) -> str:
        # No-op editor for shimming purposes.
        return code


def get_oracle_editor(config: Any = None) -> OracleCodeEditor:
    return OracleCodeEditor(config=config)


__all__ = ["OracleCodeEditor", "get_oracle_editor"]
