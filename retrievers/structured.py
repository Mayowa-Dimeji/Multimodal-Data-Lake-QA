
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
from loaders import CSVSource, DBSource, Evidence

class StructuredRetriever:
    """
    Wraps CSV + DB structured sources. Returns top-k hits per structured modality.
    """
    def __init__(self, csv_paths: List[Path], db_path: Path, table: str = "movies") -> None:
        self.csv_sources = [CSVSource(p) for p in csv_paths]
        self.db_source = DBSource(db_path, table=table)

    def search(self, query: str, k_per_modality: int = 5) -> Dict[str, List[Evidence]]:
        results: Dict[str, List[Evidence]] = {"csv": [], "db": []}
        # CSV: gather top-k from each CSV file
        csv_hits: List[Evidence] = []
        for src in self.csv_sources:
            csv_hits.extend(src.search(query, k=k_per_modality))
        # Sort by score and keep top-k overall
        csv_hits = sorted(csv_hits, key=lambda e: e.score, reverse=True)[:k_per_modality]
        results["csv"] = csv_hits

        # DB
        db_hits = self.db_source.search(query, k=k_per_modality)
        results["db"] = db_hits
        return results
