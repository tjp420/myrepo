"""Dev shim for knowledge_graph_enhancer functions referenced in api_server."""


def create_sample_knowledge_graph(*a, **k):
    return {"nodes": [], "edges": []}


__all__ = ["create_sample_knowledge_graph"]
"""Stub for core.knowledge_graph_enhancer used by api_server.

Provide a small enhancer factory used by `api_server` initialization.
"""


def enhance_graph(graph):
    return graph


class KnowledgeGraphEnhancer:
    def __init__(self, *args, **kwargs):
        pass

    def semantic_query(self, q: str):
        return {}

    def enhance_from_networkx(self, nx_graph):
        # Best-effort shim: accept networkx graphs and return a dict
        return {}

    def enhance_knowledge_graph(self, kg):
        # No-op enhancement in shim
        return kg

    @property
    def device(self):
        return getattr(self, "_device", None)

    @property
    def enhancement_config(self):
        return getattr(self, "_enhancement_config", {})


def integrate_knowledge_graph_enhancer(*args, **kwargs):
    return KnowledgeGraphEnhancer()
