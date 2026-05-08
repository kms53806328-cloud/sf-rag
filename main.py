import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = "sf-docs"

app = FastAPI(title="SF RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model...")
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded.")

pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

class Query(BaseModel):
    text:  str
    top_k: int = 5

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/context")
def context(query: Query):
    if not query.text.strip():
        raise HTTPException(status_code=400, detail="Empty query")
    
    vector  = MODEL.encode(query.text, normalize_embeddings=True).tolist()
    results = index.query(vector=vector, top_k=query.top_k, include_metadata=True)
    
    if not results.matches:
        return {"context": ""}
    
    lines = ["--- StarfallEx Reference ---"]
    for match in results.matches:
        lines.append(f"[{match.metadata.get('source', '')}]")
        if match.metadata.get('fn'):
            lines.append(match.metadata['fn'])
        lines.append(match.metadata.get('content', ''))
        lines.append("")
    lines.append("---")
    
    return {"context": "\n".join(lines)}
