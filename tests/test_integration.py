import numpy as np
from fastapi.testclient import TestClient

from unbreakable_oracle.api import app, retriever


class DummyModel:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        out = []
        for i, t in enumerate(texts):
            val = float(len(t) + i)
            out.append([val] * self.dim)
        return np.array(out, dtype=float)


def test_ingest_and_query_e2e():
    """E2E: ingest documents via API and query them."""
    # Inject a tiny deterministic dummy model to avoid heavy downloads
    retriever.model = DummyModel(dim=8)
    client = TestClient(app)

    docs = {
        "docs": [
            {"id": "doc-1", "text": "embedding retrieval test document", "meta": {}},
            {"id": "doc-2", "text": "gardening and plants info", "meta": {}},
        ]
    }

    resp = client.post("/ingest", json=docs)
    assert resp.status_code == 200, resp.text
    assert resp.json().get("ingested") == 2

    q = client.get("/query", params={"q": "embedding retrieval", "k": 2})
    assert q.status_code == 200, q.text
    data = q.json()
    assert data["query"] == "embedding retrieval"
    results = data["results"]
    assert isinstance(results, list)
    assert len(results) >= 1
    ids = [r["id"] for r in results]
    assert "doc-1" in ids
