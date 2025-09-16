
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
from loaders import Evidence

# Optional imports
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_OK = True
except Exception:
    _ST_OK = False

class UnstructuredRetriever:
    """
    Embedding-based retriever over doc chunks, using FAISS if available.
    Expects files under index_dir:
      - embeddings.npy  (shape: [N, D], float32, L2-normalized)
      - metadata.jsonl  (N lines, each with {"doc": str, "chunk": str, "source_id": str})
      - faiss.index     (optional; used if present and faiss is available)
    """
    def __init__(self, index_dir: Path, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_dir = Path(index_dir)
        self.emb_path = self.index_dir / "embeddings.npy"
        self.meta_path = self.index_dir / "metadata.jsonl"
        self.faiss_path = self.index_dir / "faiss.index"
        if not self.emb_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(f"Missing embeddings or metadata in {self.index_dir}. Run etl/build_vectors.py.")
        self.emb = np.load(self.emb_path).astype("float32")
        self.emb /= (np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-12)
        with open(self.meta_path, "r") as f:
            self.meta = [json.loads(line) for line in f]
        self.dim = self.emb.shape[1]
        self.model_name = embedding_model_name
        self._index = None
        if _FAISS_OK and self.faiss_path.exists():
            self._index = faiss.read_index(str(self.faiss_path))
        self._encoder: Optional[SentenceTransformer] = None

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not _ST_OK:
            raise RuntimeError("sentence-transformers not installed. Please install to encode queries.")
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
        vecs = self._encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        return vecs

    def search(self, query: str, k: int = 5) -> List[Evidence]:
        q = self._encode([query])
        if self._index is not None:
            D, I = self._index.search(q, k)
            idxs = I[0].tolist()
            # Using inner product; higher is better. If L2 index used, convert distance to similarity.
            scores = (1 - D[0]).tolist() if D is not None else [0.0] * len(idxs)
        else:
            sims = (self.emb @ q[0])
            idxs = np.argsort(-sims)[:k].tolist()
            scores = sims[idxs].tolist()

        hits: List[Evidence] = []
        for i, s in zip(idxs, scores):
            if i < 0 or i >= len(self.meta):
                continue
            m = self.meta[i]
            payload = {"doc": m["doc"], "snippet": m["chunk"][:500]}
            hits.append(Evidence(origin="DOC", source_id=m["source_id"], score=float(s), payload=payload))
        return hits
