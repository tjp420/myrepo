"""Stub for query_engine used by api_server."""


def execute_query(q: str):
    return None


class QueryEngine:
    def __init__(self, knowledge_graph=None):
        self.kg = knowledge_graph

    def query(self, q: str):
        return execute_query(q)
