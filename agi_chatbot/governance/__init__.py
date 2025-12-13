"""Governance helper functions exported under agi_chatbot.governance.

This module provides minimal, safe stubs used by the API server and tests.
Implementations should be provided by the real governance package in
production deployments.
"""


def get_recent_events(*args, **kwargs):
    return []


def get_event_statistics(*args, **kwargs):
    return {}


def emit_governance_event(*args, **kwargs):
    return None


def get_chain_status(*args, **kwargs):
    return None


def verify_governance_chain(*args, **kwargs):
    """Return whether the governance chain verifies successfully.

    Minimal stub: always return True (safe default for tests/dev).
    """
    return True


def list_recent_anchors(*args, **kwargs):
    return []


def list_day_roots(*args, **kwargs):
    return []


def register_governance_event_listener(*args, **kwargs):
    return None
