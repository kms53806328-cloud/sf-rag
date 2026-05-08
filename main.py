import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
HF_API_KEY       = os.environ["HF_API_KEY"]
INDEX_NAME       = "sf-docs"

HF_EMBED_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="SF RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def embed(text: str) -> list[float]:
    for _ in range(3):
        r = requests.post(
            HF_EMBED_URL,
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data[0], float):
                return data
            if isinstance(data[0], list):
                return data[0]
    raise HTTPException(status_code=502, detail="Embedding failed")

class Query(BaseModel):
    text: str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/context")
def context(query: Query):
    vector  = embed(query.text)
    results = index.query(vector=vector, top_k=query.top_k, include_metadata=True)
    if not results.matches:
        return {"context": ""}
    lines = ["--- StarfallEx Reference ---"]
    for m in results.matches:
        lines.append(f"[{m.metadata.get('source','')}]")
        if m.metadata.get('fn'):
            lines.append(m.metadata['fn'])
        lines.append(m.metadata.get('content',''))
        lines.append("")
    lines.append("---")
    return {"context": "\n".join(lines)}
