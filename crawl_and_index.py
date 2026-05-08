"""
StarfallEx RAG Crawler + Indexer
Crawls SF GitHub repo, extracts function docs, embeds and uploads to Pinecone.
"""

import os
import re
import time
import requests
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = "sf-docs"
EMBED_DIM        = 384

GITHUB_TOKEN   = os.environ.get("GITHUB_TOKEN", "")
GITHUB_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

SF_DIRS = [
    "lua/starfall/libs_sh",
    "lua/starfall/libs_cl",
    "lua/starfall/libs_sv",
]

# Load model once at startup
print("Loading embedding model...")
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded.")


# ── GitHub helpers ───────────────────────────────────────────────────────────
def gh_list(path: str) -> list:
    url = f"https://api.github.com/repos/thegrb93/StarfallEx/contents/{path}"
    r = requests.get(url, headers=GITHUB_HEADERS)
    if r.status_code != 200:
        print(f"  [skip] {path} ({r.status_code})")
        return []
    return r.json()

def gh_file(download_url: str) -> str:
    r = requests.get(download_url, headers=GITHUB_HEADERS)
    return r.text if r.status_code == 200 else ""

def collect_lua_files(path: str) -> list[tuple[str, str]]:
    """Recursively collect (name, content) for all .lua files under path."""
    results = []
    items = gh_list(path)
    for item in items:
        if item["type"] == "file" and item["name"].endswith(".lua"):
            print(f"  fetching {item['path']}")
            content = gh_file(item["download_url"])
            results.append((item["path"], content))
            time.sleep(0.2)  # be polite to GitHub API
        elif item["type"] == "dir":
            results.extend(collect_lua_files(item["path"]))
    return results

# ── Doc parser ───────────────────────────────────────────────────────────────
def parse_chunks(filepath: str, source: str) -> list[dict]:
    """
    Extract documented functions from a Lua source file.
    Looks for blocks like:
        --- Description
        -- @param ...
        -- @return ...
        function lib.name(...)
    """
    chunks = []
    lines  = source.splitlines()
    i      = 0

    while i < len(lines):
        line = lines[i]
        # Start of a doc comment block
        if line.strip().startswith("---"):
            doc_lines = []
            while i < len(lines) and (lines[i].strip().startswith("---") or lines[i].strip().startswith("-- @")):
                doc_lines.append(lines[i].strip().lstrip("- ").strip())
                i += 1
            # Look for the function signature on the next non-empty line
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i < len(lines):
                fn_line = lines[i].strip()
                if "function" in fn_line:
                    content = fn_line + "\n" + "\n".join(doc_lines)
                    chunks.append({
                        "content": content,
                        "source":  filepath,
                        "fn":      fn_line[:120],
                    })
        i += 1

    # Fallback: if no doc blocks found, chunk the whole file into 400-char windows
    if not chunks:
        text = source
        size = 400
        for j in range(0, len(text), size):
            chunk = text[j:j+size].strip()
            if chunk:
                chunks.append({
                    "content": chunk,
                    "source":  filepath,
                    "fn":      "",
                })

    return chunks

# ── Embedding ────────────────────────────────────────────────────────────────
def embed_batch(texts: list[str]) -> list[list[float]]:
    vectors = MODEL.encode(texts, normalize_embeddings=True)
    return vectors.tolist()

# ── Pinecone upsert ──────────────────────────────────────────────────────────
def upsert_chunks(index, chunks: list[dict], batch_size: int = 50):
    texts = [c["content"] for c in chunks]
    # Embed in sub-batches (HF API limit)
    vectors = []
    sub = 16
    for i in range(0, len(texts), sub):
        print(f"  embedding {i}–{min(i+sub, len(texts))} / {len(texts)}")
        batch_vecs = embed_batch(texts[i:i+sub])
        vectors.extend(batch_vecs)
        time.sleep(0.5)

    # Upsert to Pinecone in batches
    records = []
    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
        if all(v == 0.0 for v in vec):  # skip zero vectors
            continue
        records.append({
            "id":       f"{chunk['source']}_{idx}",
            "values":   vec,
            "metadata": {
                "content": chunk["content"][:1000],
                "source":  chunk["source"],
                "fn":      chunk["fn"],
            },
        })

    for i in range(0, len(records), batch_size):
        index.upsert(vectors=records[i:i+batch_size])
        print(f"  upserted {i}–{min(i+batch_size, len(records))} / {len(records)}")
        time.sleep(0.3)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # Init Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(5)

    index = pc.Index(INDEX_NAME)
    print(f"Pinecone index '{INDEX_NAME}' ready.")

    # Crawl SF repo
    all_files = []
    for sf_dir in SF_DIRS:
        print(f"\nCrawling {sf_dir}...")
        all_files.extend(collect_lua_files(sf_dir))

    print(f"\nTotal files fetched: {len(all_files)}")

    # Parse + index
    total_chunks = 0
    for filepath, content in all_files:
        if not content.strip():
            continue
        chunks = parse_chunks(filepath, content)
        if not chunks:
            continue
        print(f"\nIndexing {filepath} ({len(chunks)} chunks)")
        upsert_chunks(index, chunks)
        total_chunks += len(chunks)

    print(f"\nDone. Total chunks indexed: {total_chunks}")

if __name__ == "__main__":
    main()
