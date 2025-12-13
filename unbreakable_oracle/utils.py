"""Utilities for the Unbreakable Oracle RAG prototype.

Contains simple text cleaning/chunking helpers and wrappers around
SentenceTransformers embeddings (loaded lazily).
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """Normalize whitespace and trim."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split `text` into chunks of size `chunk_size` with `overlap`.

    This is a simple sliding-window chunker useful for indexing long documents.
    """
    text = clean_text(text)
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load and return a SentenceTransformer model instance.

    Raises RuntimeError if `sentence-transformers` is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - dependency error
        raise RuntimeError(
            "`sentence-transformers` not installed. Install with `pip install sentence-transformers`."
        ) from exc
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model=None, model_name: str = "all-MiniLM-L6-v2"):
    """Return embeddings for a list of `texts` as a NumPy array.

    If `model` is None this will lazily load SentenceTransformer with `model_name`.
    """
    import numpy as _np

    if model is None:
        model = load_sentence_transformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if not isinstance(embeddings, _np.ndarray):
        embeddings = _np.array(embeddings)
    return embeddings


def normalize_embeddings(embeddings):
    """L2-normalize embeddings (rows) to unit length."""
    import numpy as _np

    if embeddings is None:
        return embeddings
    norms = _np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = _np.where(norms == 0, 1e-12, norms)
    return embeddings / norms
