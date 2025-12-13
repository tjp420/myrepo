# Unbreakable Oracle — RAG Prototype

This folder contains a small Retrieval-Augmented-Generation (RAG) prototype:
- `unbreakable_oracle.api` — FastAPI app with endpoints to ingest docs, query, and a tiny KG.
- `unbreakable_oracle.retriever` — SentenceTransformers embeddings + FAISS (optional) retriever.
- `scripts/ingest_folder.py` — simple script to index a folder of text/markdown files.

Requirements
- Use the minimal requirements for the prototype:

```sh
pip install -r unbreakable_oracle/requirements.txt
```

Quickstart (virtualenv)

```sh
python -m venv .venv
.venv\Scripts\activate    # Windows
# or
source .venv/bin/activate  # macOS/Linux
pip install -r unbreakable_oracle/requirements.txt
uvicorn unbreakable_oracle.api:app --reload
```

Example usage

Ingest documents via API:

```sh
curl -X POST "http://localhost:8000/ingest" -H "Content-Type: application/json" \
  -d '{"docs": [{"text": "This is a test document about AI and retrieval."}] }'
```

Query the index:

```sh
curl "http://localhost:8000/query?q=AI&k=3"
```

Index a folder of files (local script):

```sh
python scripts/ingest_folder.py path/to/docs
```

Docker

Build and run the prototype container (build from repo root):

```sh
docker build -f unbreakable_oracle/Dockerfile -t unbreakable_oracle:local .
docker run --rm -p 8000:8000 unbreakable_oracle:local
```

Notes
- `sentence-transformers` downloads model weights on first run (internet access required).
- `faiss-cpu` is optional (faster nearest-neighbor search). On Windows it may be easier to use conda to install FAISS.
- This is a minimal prototype intended for experimentation. For production use: persist indexes to disk, implement incremental indexing, add authentication, rate limits, logging, and unit/integration tests.

Next steps
- Add ingestion persistence and incremental indexing.
- Add tests and a GitHub Actions workflow.
- Add example dataset and evaluation harness.
