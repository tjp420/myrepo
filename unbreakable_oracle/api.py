"""FastAPI app for the Unbreakable Oracle RAG prototype.

Endpoints:
- POST /ingest  -> ingest documents and build index
- GET  /query   -> query the vector store
- POST /kg/entity -> add entity to KG
- GET  /kg/entity/{id} -> fetch entity
- POST /kg/edge -> add relation

Run with: `uvicorn unbreakable_oracle.api:app --reload`
"""

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .kg import KnowledgeGraph
from .retriever import Retriever

app = FastAPI(title="Unbreakable Oracle RAG Prototype")

# instantiate prototype components (models are lazy-loaded)
retriever = Retriever()
kg = KnowledgeGraph()


class DocModel(BaseModel):
    id: Optional[str] = None
    text: str
    meta: Optional[dict] = None


class IngestRequest(BaseModel):
    docs: List[DocModel]


class KGEntity(BaseModel):
    id: str
    attributes: Optional[dict] = None


class KGRelation(BaseModel):
    source: str
    target: str
    relation: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    docs = []
    for i, d in enumerate(req.docs):
        doc_id = d.id or f"doc-{i}"
        docs.append({"id": doc_id, "text": d.text, "meta": d.meta or {}})

    count = retriever.build_index(docs)
    return {"ingested": count}


@app.get("/query")
def query(q: str, k: int = 5):
    if not q:
        raise HTTPException(status_code=400, detail="`q` query param required")
    results = retriever.query(q, k=k)
    return {"query": q, "results": results}


@app.post("/kg/entity")
def add_entity(entity: KGEntity):
    kg.add_entity(entity.id, entity.attributes or {})
    return {"ok": True}


@app.get("/kg/entity/{eid}")
def get_entity(eid: str):
    e = kg.get_entity(eid)
    if e is None:
        raise HTTPException(status_code=404, detail="entity not found")
    return {"id": eid, "attributes": e}


@app.post("/kg/edge")
def add_edge(rel: KGRelation):
    kg.add_edge(rel.source, rel.target, rel.relation)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("unbreakable_oracle.api:app", host="0.0.0.0", port=8000, reload=True)
