import numpy as np

from unbreakable_oracle.retriever import Retriever


class DummyModel:
    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        out = []
        for i, t in enumerate(texts):
            val = float(len(t) + i)
            out.append([val] * self.dim)
        return np.array(out, dtype=float)


def test_sharded_save_and_load(tmp_path):
    r = Retriever()
    r.model = DummyModel(dim=8)
    docs = [{"id": f"d{i}", "text": "word" * (i + 1), "meta": {}} for i in range(5)]
    r.build_index(docs)

    outdir = tmp_path / "idx_sharded"
    # force shard size small to create multiple shards
    r.save(outdir, shard_size=2)

    r2 = Retriever()
    r2.load(outdir)

    assert len(r2.docs) == len(r.docs)
    assert r2.embeddings is not None
    assert r2.embeddings.shape == r.embeddings.shape
