"""Stub for context.context_window_manager used in api_server."""


def get_window_manager(*args, **kwargs):
    return None


def get_context_window_manager(*args, **kwargs):
    return get_window_manager(*args, **kwargs)
