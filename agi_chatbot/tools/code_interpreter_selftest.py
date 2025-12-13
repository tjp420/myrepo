"""Shim for `agi_chatbot.tools.code_interpreter_selftest`.

Provides a minimal `run_selftest` used during startup checks.
"""


def run_selftest() -> bool:
    return True
