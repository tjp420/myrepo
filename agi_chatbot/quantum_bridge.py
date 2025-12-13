"""Minimal quantum_bridge shim for linting compatibility.

Expose `get_quantum_bridge` used by `api_server`.
"""


def bridge_call(*args, **kwargs):
    return None


def get_quantum_bridge(*args, **kwargs):
    return bridge_call
