"""Lightweight NLP shim used during dev/test.

Provides a minimal `NLPTokenizer` used by the API server import chain.
"""

from typing import List


class NLPTokenizer:
    def __init__(self, model: str = "simple") -> None:
        self.model = model

    def tokenize(self, text: str) -> List[str]:
        # Very small tokenizer suitable for smoke tests.
        if not text:
            return []
        return text.split()


__all__ = ["NLPTokenizer"]
