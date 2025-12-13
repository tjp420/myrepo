"""Simple circuit breaker shim for dev/test.

Exports `get_circuit_breaker` and `CircuitBreakerError` used by api_server.
"""

from typing import Any, Callable


class CircuitBreakerError(Exception):
    pass


class CircuitBreaker:
    def __init__(self) -> None:
        pass

    def call(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        # Directly invoke the function in the shim.
        return fn(*args, **kwargs)


def get_circuit_breaker(*args, **kwargs) -> CircuitBreaker:
    # Accept arbitrary keyword arguments used by the real implementation
    # (e.g. name, exceptions, timeout) and return a simple shim instance.
    return CircuitBreaker()


__all__ = ["CircuitBreaker", "CircuitBreakerError", "get_circuit_breaker"]
