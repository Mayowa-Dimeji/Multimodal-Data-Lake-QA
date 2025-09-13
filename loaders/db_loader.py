from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import List
from .common import Evidence

class DBSource:
    def __init__(self, db_path: Path, table: str = "movies", key_column: str = "title") -> None:
        self.db_path = Path(db_path)
        self.table = table
        self.key_column = key_column

    def search(self, query: str, k: int = 5) -> List[Evidence]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        # Basic LIKE search. In Step 2 we'll improve with embeddings, etc.
        sql = f"SELECT * FROM {self.table} WHERE {self.key_column} LIKE ? LIMIT ?"
        rows = con.execute(sql, (f"%{query}%", k)).fetchall()
        con.close()
        hits = []
        for r in rows:
            row = dict(r)
            source_id = f"db:{self.table}:{row.get(self.key_column,'row')}"
            hits.append(Evidence(origin='DB', source_id=source_id, score=1.0, payload=row))
        return hits
