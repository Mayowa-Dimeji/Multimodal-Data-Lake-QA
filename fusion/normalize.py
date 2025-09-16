
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import re

def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def canonical_title(raw_title: str, release_year: int | None = None) -> str:
    if not raw_title:
        return ""
    t = _canon(raw_title).strip('"\'')
    return f"{t} ({release_year})" if release_year else t

def row_to_triples(row: Dict[str, Any], subject_hint: str = "movie") -> List[Tuple[str, str, Any]]:
    triples = []
    for k, v in row.items():
        if v is None or v == "":
            continue
        triples.append((subject_hint, k, v))
    return triples

def normalize_retrieval(query: str, retrieval: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out = {"query": query, "retrieval": {"db": [], "csv": [], "docs": []}, "entities": {"canonical_map": {}}}
    canonical_map: Dict[str, str] = {}

    for h in retrieval.get("db", []):
        row = dict(h["payload"])
        title = row.get("title", "")
        year = row.get("release_year") or row.get("year")
        can = canonical_title(title, year if isinstance(year, int) else None)
        if title:
            canonical_map[title] = can
        out["retrieval"]["db"].append({
            "source_id": h["source_id"],
            "origin": "DB",
            "table": "movies",
            "row": row,
            "triples": row_to_triples(row),
            "canonical_id": can,
            "score": float(h.get("score", 0.0)),
        })

    for h in retrieval.get("csv", []):
        row = dict(h["payload"])
        title = row.get("title", "")
        year = row.get("release_year") or row.get("year")
        can = canonical_title(title, year if isinstance(year, int) else None)
        if title:
            canonical_map.setdefault(title, can)
        out["retrieval"]["csv"].append({
            "source_id": h["source_id"],
            "origin": "CSV",
            "file": "movies.csv" if "imdb" not in row else "ratings.csv",
            "row": row,
            "triples": row_to_triples(row),
            "canonical_id": can,
            "score": float(h.get("score", 0.0)),
        })

    for h in retrieval.get("docs", []):
        payload = dict(h["payload"])
        doc = payload.get("doc", "")
        snippet = payload.get("snippet", "")
        out["retrieval"]["docs"].append({
            "source_id": h["source_id"],
            "origin": "DOC",
            "chunk": snippet,
            "metadata": {"doc": doc},
            "score": float(h.get("score", 0.0)),
        })

    out["entities"]["canonical_map"] = canonical_map
    return out
