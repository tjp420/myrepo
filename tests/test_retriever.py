import numpy as np

from unbreakable_oracle.retriever import Retriever


class DummyModel:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        # produce deterministic embeddings: vector of (len(text) + index)
        out = []
        for i, t in enumerate(texts):
            val = float(len(t) + i)
            out.append([val] * self.dim)
        return np.array(out, dtype=float)


def test_retriever_basic():
    r = Retriever()
    # inject dummy model to avoid downloading heavy models during tests
    r.model = DummyModel(dim=8)
    docs = [
        {"id": "d1", "text": "hello world", "meta": {}},
        {"id": "d2", "text": "another document", "meta": {}},
        {"id": "d3", "text": "short", "meta": {}},
    ]
    assert r.build_index(docs) == 3
    results = r.query("test query", k=2)
    assert len(results) == 2
    assert all("id" in r and "score" in r for r in results)
    scores = [r["score"] for r in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
