"""A tiny in-memory knowledge graph for demonstration and tests."""

from typing import Any, Dict, List, Optional


class KnowledgeGraph:
    """Simple graph storing entities and typed edges.

    Entities are stored as a mapping id -> attributes(dict). Edges are stored as
    (source, target, relation) records.
    """

    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, str]] = []

    def add_entity(self, eid: str, attributes: Optional[Dict[str, Any]] = None):
        self.entities[eid] = attributes or {}
        return True

    def get_entity(self, eid: str) -> Optional[Dict[str, Any]]:
        return self.entities.get(eid)

    def add_edge(self, source: str, target: str, relation: str):
        self.edges.append({"source": source, "target": target, "relation": relation})
        return True

    def neighbors(self, eid: str) -> List[str]:
        out = []
        for e in self.edges:
            if e["source"] == eid:
                out.append(e["target"])
            if e["target"] == eid:
                out.append(e["source"])
        return out

    def query_by_attr(self, key: str, value: Any) -> List[str]:
        """Return entity ids where attributes[key] == value."""
        matches: List[str] = []
        for eid, attrs in self.entities.items():
            if isinstance(attrs, dict) and attrs.get(key) == value:
                matches.append(eid)
        return matches
