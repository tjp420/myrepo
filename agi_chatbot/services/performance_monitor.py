"""Lightweight services.performance_monitor shim for linting/tests."""


class PerformanceMonitor:
    def record_request(self, *args, **kwargs):
        return None


def get_performance_monitor():
    return PerformanceMonitor()
