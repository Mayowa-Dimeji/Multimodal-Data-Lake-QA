from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List
from rapidfuzz import fuzz
from .common import Evidence

class CSVSource:
    def __init__(self, file_path: Path, key_column: str = "title") -> None:
        self.file_path = Path(file_path)
        self.key_column = key_column
        self.df = pd.read_csv(self.file_path)

    def search(self, query: str, k: int = 5) -> List[Evidence]:
        # Simple fuzzy ratio on the key column
        scores = []
        for idx, row in self.df.iterrows():
            title = str(row.get(self.key_column, ""))
            score = fuzz.token_set_ratio(query, title) / 100.0
            scores.append((score, idx))
        scores.sort(reverse=True)
        hits = []
        for score, idx in scores[:k]:
            payload = self.df.iloc[idx].to_dict()
            source_id = f"csv:{self.file_path.name}:{payload.get(self.key_column,'row_'+str(idx))}"
            hits.append(Evidence(origin="CSV", source_id=source_id, score=score, payload=payload))
        return hits
