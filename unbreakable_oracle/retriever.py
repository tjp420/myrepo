
"""Retriever: SentenceTransformers + FAISS (with numpy fallback).

Simple prototype of a vector retriever that supports building an index from
in-memory documents and querying by similarity.
"""
from typing import List, Dict, Optional, Union
import json
from pathlib import Path

import numpy as np

from .utils import embed_texts, normalize_embeddings, load_sentence_transformer


class Retriever:
    """A minimal retriever for prototyping RAG-style workflows.

    - Uses `sentence-transformers` for embeddings.
    - Uses FAISS `IndexFlatIP` if available, otherwise falls back to numpy brute-force.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.docs: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._faiss = False
        self.index = None
        try:
            import faiss  # type: ignore

            self._faiss = True
            self._faiss_module = faiss
        except Exception:  # pragma: no cover - optional dependency
            self._faiss = False
            self._faiss_module = None

    def _get_model(self):
        if self.model is None:
            self.model = load_sentence_transformer(self.model_name)
        return self.model

    def build_index(self, docs: List[Dict]):
        if not docs:
            self.docs = []
            self.embeddings = None
            self.index = None
            return 0

        texts = [d.get("text", "") for d in docs]
        model = self._get_model()
        embs = embed_texts(texts, model=model, model_name=self.model_name)
        embs = normalize_embeddings(embs).astype("float32")

        self.docs = docs
        self.embeddings = embs

        if self._faiss:
            dim = embs.shape[1]
            self.index = self._faiss_module.IndexFlatIP(dim)
            self.index.add(embs)
        else:
            self.index = None

        return len(self.docs)

    def query(self, query_text: str, k: int = 5) -> List[Dict]:
        if not self.docs:
            return []
        model = self._get_model()
        q_emb = embed_texts([query_text], model=model, model_name=self.model_name)
        q_emb = normalize_embeddings(q_emb).astype("float32")

        if self._faiss and self.index is not None:
            distances, indices = self.index.search(q_emb, k)
            distances = distances.tolist()[0]
            indices = indices.tolist()[0]
        else:
            sims = (self.embeddings @ q_emb.T).flatten()
            idx = np.argsort(-sims)[:k]
            indices = idx.tolist()
            distances = sims[idx].tolist()

        results = []
        for i, score in zip(indices, distances):
            if i < 0 or i >= len(self.docs):
                continue
            doc = self.docs[i].copy()
            doc["score"] = float(score)
            results.append(doc)
        return results

    def add_and_index(self, doc: Union[Dict, List[Dict]]):
        docs_to_add: List[Dict]
        if isinstance(doc, dict):
            docs_to_add = [doc]
        else:
            docs_to_add = list(doc)

        if not docs_to_add:
            return 0

        model = self._get_model()
        texts = [d.get("text", "") for d in docs_to_add]
        new_embs = embed_texts(texts, model=model, model_name=self.model_name)
        new_embs = normalize_embeddings(new_embs).astype("float32")

        start_len = len(self.docs)
        self.docs.extend(docs_to_add)

        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

        if self._faiss:
            if self.index is None:
                dim = self.embeddings.shape[1]
                self.index = self._faiss_module.IndexFlatIP(dim)
                self.index.add(self.embeddings)
            else:
                self.index.add(new_embs)
        else:
            self.index = None

        return len(self.docs) - start_len

    def save(self, path: Union[str, Path], shard_size: Optional[int] = None):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        n_docs = len(self.docs) if self.docs is not None else 0

        if shard_size and shard_size > 0 and n_docs > shard_size:
            shard_count = (n_docs + shard_size - 1) // shard_size
            shards = []
            for i in range(shard_count):
                start = i * shard_size
                end = min(start + shard_size, n_docs)
                docs_shard = self.docs[start:end]
                emb_shard = None
                if self.embeddings is not None:
                    emb_shard = self.embeddings[start:end]

                shard_name = f"shard_{i}"
                shard_dir = p / shard_name
                shard_dir.mkdir(parents=True, exist_ok=True)

                with open(shard_dir / "docs.json", "w", encoding="utf-8") as fh:
                    json.dump(docs_shard or [], fh, ensure_ascii=False)

                if emb_shard is not None:
                    np.save(shard_dir / "embeddings.npy", emb_shard.astype("float32"))
                    if self._faiss:
                        try:
                            dim = int(emb_shard.shape[1])
                            idx = self._faiss_module.IndexFlatIP(dim)
                            idx.add(emb_shard)
                            self._faiss_module.write_index(idx, str(shard_dir / "index.faiss"))
                        except Exception:
                            pass

                shards.append(shard_name)

            with open(p / "shards.json", "w", encoding="utf-8") as fh:
                json.dump({"shards": shards}, fh, ensure_ascii=False)

            return True

        with open(p / "docs.json", "w", encoding="utf-8") as fh:
            json.dump(self.docs or [], fh, ensure_ascii=False)

        if self.embeddings is not None:
            np.save(p / "embeddings.npy", self.embeddings.astype("float32"))

        if self._faiss and (self.index is not None):
            try:
                self._faiss_module.write_index(self.index, str(p / "index.faiss"))
            except Exception:
                pass

    def load(self, path: Union[str, Path]):
        p = Path(path)
        docs_file = p / "docs.json"
        if docs_file.exists():
            with open(docs_file, "r", encoding="utf-8") as fh:
                self.docs = json.load(fh)
        else:
            self.docs = []

        emb_file = p / "embeddings.npy"
        if emb_file.exists():
            self.embeddings = np.load(str(emb_file))
        else:
            self.embeddings = None

        shards_file = p / "shards.json"
        if shards_file.exists():
            meta = json.load(open(shards_file, "r", encoding="utf-8"))
            shards = meta.get("shards", [])
            loaded_docs = []
            embs_list = []
            for s in shards:
                shard_dir = p / s
                docs_file = shard_dir / "docs.json"
                emb_file = shard_dir / "embeddings.npy"
                if docs_file.exists():
                    with open(docs_file, "r", encoding="utf-8") as fh:
                        loaded_docs.extend(json.load(fh))
                if emb_file.exists():
                    embs_list.append(np.load(str(emb_file)))

            self.docs = loaded_docs
            if embs_list:
                self.embeddings = np.vstack(embs_list)
            else:
                self.embeddings = None

            if self._faiss and (self.embeddings is not None):
                dim = int(self.embeddings.shape[1])
                self.index = self._faiss_module.IndexFlatIP(dim)
                self.index.add(self.embeddings.astype("float32"))
            else:
                self.index = None
            return

        if self._faiss:
            idx_file = p / "index.faiss"
            if idx_file.exists():
                try:
                    self.index = self._faiss_module.read_index(str(idx_file))
                except Exception:
                    self.index = None
            if (self.index is None) and (self.embeddings is not None):
                dim = int(self.embeddings.shape[1])
                self.index = self._faiss_module.IndexFlatIP(dim)
                self.index.add(self.embeddings.astype("float32"))
        else:
            self.index = None
                            idx_file = p / "index.faiss"
                            if idx_file.exists():
                                try:
                                    self.index = self._faiss_module.read_index(
                                        str(idx_file)
                                    )
                                except Exception:
                                    self.index = None
                            if (self.index is None) and (self.embeddings is not None):
                                dim = int(self.embeddings.shape[1])
                                self.index = self._faiss_module.IndexFlatIP(dim)
                                self.index.add(self.embeddings.astype("float32"))
                        else:
                            self.index = None
