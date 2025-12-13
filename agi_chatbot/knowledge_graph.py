"""Stub for knowledge_graph used by api_server."""


def get_graph(name: str = None):
    return {}


class KnowledgeGraph:
    def __init__(self, *args, **kwargs):
        pass

    def add_node(self, node):
        return None

    def query(self, q: str):
        return {}

    def get_user_context(self, user_id: str):
        return {}

    def get_recommendations(self, user_id: str):
        return []
