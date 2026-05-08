"""
StarfallEx RAG Search API
Receives a query, embeds it, searches Pinecone, returns relevant SF doc chunks.
"""

import os
import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone

# ── Config ───────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
HF_API_KEY       = os.environ["HF_API_KEY"]
INDEX_NAME       = "sf-docs"
EMBED_DIM        = 384

HF_EMBED_URL = (
    "https://api-inference.huggingface.co/pipeline/feature-extraction/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ── Init ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="SF RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ── Embedding ────────────────────────────────────────────────────────────────
def embed(text: str, retries: int = 3) -> list[float]:
    for attempt in range(retries):
        try:
            r = requests.post(
                HF_EMBED_URL,
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": text, "options": {"wait_for_model": True}},
                timeout=30,
            )
            data = r.json()
            if isinstance(data, list) and isinstance(data[0], float):
                return data          # single text → flat list
            if isinstance(data, list) and isinstance(data[0], list):
                return data[0]       # batched response
        except Exception as e:
            print(f"embed error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    raise HTTPException(status_code=502, detail="Embedding service unavailable")

# ── Routes ───────────────────────────────────────────────────────────────────
class Query(BaseModel):
    text:  str
    top_k: int = 5

class SearchResult(BaseModel):
    score:   float
    content: str
    source:  str
    fn:      str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search", response_model=list[SearchResult])
def search(query: Query):
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    vector  = embed(query.text)
    results = index.query(
        vector=vector,
        top_k=min(query.top_k, 10),
        include_metadata=True,
    )

    return [
        SearchResult(
            score   = round(match.score, 4),
            content = match.metadata.get("content", ""),
            source  = match.metadata.get("source", ""),
            fn      = match.metadata.get("fn", ""),
        )
        for match in results.matches
    ]

@app.post("/context")
def context(query: Query):
    """
    Returns a single formatted string ready to inject into an LLM prompt.
    """
    results = search(query)
    if not results:
        return {"context": ""}

    lines = ["--- StarfallEx Reference (retrieved) ---"]
    for r in results:
        lines.append(f"[{r.source}]")
        if r.fn:
            lines.append(r.fn)
        lines.append(r.content)
        lines.append("")
    lines.append("---")

    return {"context": "\n".join(lines)}
