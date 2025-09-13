from __future__ import annotations
from pathlib import Path
from typing import List
from rapidfuzz import fuzz
from .common import Evidence

class DocSource:
    def __init__(self, dir_path: Path) -> None:
        self.dir_path = Path(dir_path)
        self.docs = []
        for p in self.dir_path.glob("*.txt"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            self.docs.append((p.name, text))

    def search(self, query: str, k: int = 5) -> List[Evidence]:
        scored = []
        for name, text in self.docs:
            score = fuzz.partial_ratio(query, text) / 100.0
            scored.append((score, name, text))
        scored.sort(reverse=True)
        hits = []
        for score, name, text in scored[:k]:
            snippet = text[:280]
            payload = {"doc": name, "snippet": snippet}
            source_id = f"doc:{name}"
            hits.append(Evidence(origin="DOC", source_id=source_id, score=score, payload=payload))
        return hits
