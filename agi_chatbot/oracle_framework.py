"""Dev stubs for the oracle framework types referenced in `api_server`.

This module provides lightweight, import-safe stubs used during
development and testing to satisfy type and runtime imports from the
application. Implementations here are intentionally minimal and should
be replaced by the production oracle implementation in real deployments.
"""

from typing import Any, Dict, List


class Oracle:
    def __init__(self) -> None:
        self.sources: List[Dict[str, Any]] = []

    def ask(self, *args, **kwargs) -> Any:
        return None

    def add_source(self, src: Dict[str, Any]) -> None:
        self.sources.append(src)

    def query(self, query_text: str, *args, **kwargs) -> Any:
        return None

    def get_summary(self) -> Dict[str, Any]:
        return {"sources": len(self.sources)}


class SourceType:
    WEB = "web"
    DB = "db"
    ACADEMIC = "academic"
    GOVERNMENT = "government"
    NEWS = "news"


class Question:
    def __init__(self, text: str = ""):
        self.text = text


class OracleFramework:
    def query(self, q: str) -> Any:
        return None


def get_oracle_framework() -> OracleFramework:
    return OracleFramework()


__all__ = [
    "Oracle",
    "SourceType",
    "Question",
    "OracleFramework",
    "get_oracle_framework",
]
