"""Stub for core.temporal_verification."""


def verify_temporal_constraints(*args, **kwargs):
    return True


class TemporalVerificationEngine:
    def verify(self, *args, **kwargs):
        return verify_temporal_constraints(*args, **kwargs)


def get_temporal_verification_engine():
    return TemporalVerificationEngine()
