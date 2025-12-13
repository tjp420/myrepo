"""Dev shim: advanced analytics entrypoint.

Exports `get_analytics_engine` used by the API in dev mode.
"""

from typing import Any


class _AnalyticsEngine:
    def get_stats(self):
        return {"uptime": 0, "events": 0}


def get_analytics_engine() -> Any:
    return _AnalyticsEngine()


"""Stub for core.advanced_analytics used by api_server."""


def compute_analytics(data):
    return {}
