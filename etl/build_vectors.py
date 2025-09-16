
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer # type: ignore
    _ST_OK = True
except Exception:
    _ST_OK = False

BASE = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE / "data_lake" / "docs"
INDEX_DIR = BASE / "indexes" / "docs"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks() -> List[Dict]:
    chunks = []
    for p in DOCS_DIR.glob("*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        chunks.append({"doc": p.name, "chunk": text, "source_id": f"doc:{p.name}"})
    return chunks

def main():
    if not _ST_OK:
        raise RuntimeError("sentence-transformers not installed. Please `pip install sentence-transformers torch`.")
    encoder = SentenceTransformer(MODEL_NAME)
    chunks = load_chunks()
    texts = [c["chunk"] for c in chunks]
    X = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    np.save(INDEX_DIR / "embeddings.npy", X)
    with open(INDEX_DIR / "metadata.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    if _FAISS_OK:
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
        print(f"Built FAISS index with {X.shape[0]} vectors at {INDEX_DIR/'faiss.index'}")
    else:
        print("FAISS not installed; using NumPy search fallback.")
    print(f"Saved embeddings to {INDEX_DIR/'embeddings.npy'} and metadata to {INDEX_DIR/'metadata.jsonl'}")

if __name__ == "__main__":
    main()
