"""Unbreakable Oracle RAG prototype package.

This package contains a minimal FastAPI app, a retriever (SentenceTransformers + FAISS
fallback), a tiny in-memory knowledge graph, and utility helpers.
"""

__all__ = ["api", "retriever", "kg", "utils"]
