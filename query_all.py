
from pathlib import Path
from retrievers import UnifiedRetriever

BASE = Path(__file__).resolve().parent
csv_paths = [
    BASE / "data_lake" / "csv" / "movies.csv",
    BASE / "data_lake" / "csv" / "ratings.csv",
]
db_path = BASE / "data_lake" / "db" / "movies.db"
docs_index = BASE / "indexes" / "docs"

def show(label, hits):
    print(f"\n== {label} ==")
    for h in hits:
        if h.origin in ("CSV","DB"):
            summary = {k: v for k, v in h.payload.items() if k in ("title","release_year","box_office_usd","imdb","metacritic")}
        else:
            summary = (h.payload.get("snippet","")[:120] + "...").replace("\n"," ")
        print(f"- [{h.origin}] {h.source_id} score={h.score:.3f} :: {summary}")

def main():
    retr = UnifiedRetriever(csv_paths=csv_paths, db_path=db_path, docs_index_dir=docs_index)
    q = "mind-bending dreams in Nolan films"
    out = retr.search_all(q, k_per_modality=5)
    show("CSV", out["csv"])
    show("DB", out["db"])
    show("DOCS", out["docs"])

if __name__ == "__main__":
    main()
