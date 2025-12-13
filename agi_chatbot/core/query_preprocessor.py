"""Stub for core.query_preprocessor used by api_server."""


def preprocess_query(query: str) -> str:
    return query


def get_query_preprocessor():
    return preprocess_query
