"""Stub for core.performance_monitor to satisfy imports."""


class CorePerformanceMonitor:
    def record(self, *args, **kwargs):
        return None


def get_performance_monitor():
    return CorePerformanceMonitor()
