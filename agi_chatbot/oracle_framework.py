"""Dev stubs for the oracle framework types referenced in api_server."""

from typing import Any


class Oracle:
    def __init__(self) -> None:
        self.sources: List[Dict[str, Any]] = []

    def ask(self, *a, **k) -> Any:
        return None

    def add_source(self, src: Dict[str, Any]) -> None:
        self.sources.append(src)

    def query(self, query_text: str, *a, **k) -> Any:
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


__all__ = ["Oracle", "SourceType", "Question"]
"""Stub for agi_chatbot.oracle_framework referenced by api_server."""


class OracleFramework:
    def query(self, q):
        return None


def get_oracle_framework():
    return OracleFramework()
