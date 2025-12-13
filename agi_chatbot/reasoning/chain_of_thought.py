"""Chain-of-thought reasoning shim for linting."""


def generate_steps(prompt: str):
    return []


class ChainOfThoughtEngine:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt: str):
        return generate_steps(prompt)
