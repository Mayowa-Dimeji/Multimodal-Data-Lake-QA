
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
from loaders import Evidence
from .structured import StructuredRetriever
from .unstructured import UnstructuredRetriever

class UnifiedRetriever:
    def __init__(self, csv_paths: List[Path], db_path: Path, docs_index_dir: Path) -> None:
        self.structured = StructuredRetriever(csv_paths=csv_paths, db_path=db_path)
        self.unstructured = UnstructuredRetriever(index_dir=docs_index_dir)

    def search_all(self, query: str, k_per_modality: int = 5) -> Dict[str, List[Evidence]]:
        out: Dict[str, List[Evidence]] = {"csv": [], "db": [], "docs": []}
        struct = self.structured.search(query, k_per_modality=k_per_modality)
        out["csv"] = struct.get("csv", [])
        out["db"] = struct.get("db", [])
        out["docs"] = self.unstructured.search(query, k=k_per_modality)
        return out
