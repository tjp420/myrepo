"""Shim for agi_chatbot.rate_limiting_utils to satisfy imports during linting/tests."""


class _NoopLimiter:
    def allow(self, *args, **kwargs):
        return True

    def set_domain_limit(self, *args, **kwargs):
        return None

    def set_user_limit(self, *args, **kwargs):
        return None


def get_domain_rate_limiter(*args, **kwargs):
    return _NoopLimiter()
