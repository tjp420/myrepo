"""Stub for ethics_safety.ethics_safety_module."""


def evaluate_ethics(prompt: str):
    return True


class EthicsSafetyModule:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, prompt: str):
        return evaluate_ethics(prompt)
