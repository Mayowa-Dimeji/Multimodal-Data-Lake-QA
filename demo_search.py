from pathlib import Path
from loaders import CSVSource, DocSource, DBSource

BASE = Path(__file__).resolve().parent
csv_movies = BASE / "data_lake" / "csv" / "movies.csv"
csv_ratings = BASE / "data_lake" / "csv" / "ratings.csv"
docs_dir = BASE / "data_lake" / "docs"
db_path = BASE / "data_lake" / "db" / "movies.db"

def pretty(hits):
    for h in hits:
        print(f"- [{h.origin}] {h.source_id} (score={h.score:.2f})")
        if h.origin in ("CSV", "DB"):
            print("  ", {k: v for k, v in h.payload.items() if k in ("title","release_year","box_office_usd","imdb","metacritic")})
        else:
            print("  ", h.payload["snippet"].replace("\n", " ")[:120], "...")

def main():
    print("== CSV: movies.csv ==")
    csv_src = CSVSource(csv_movies)
    hits = csv_src.search("Inception")
    pretty(hits)

    print("\n== CSV: ratings.csv ==")
    csv_r = CSVSource(csv_ratings)
    hits = csv_r.search("Interstellar")
    pretty(hits)

    print("\n== DOCS ==")
    doc_src = DocSource(docs_dir)
    hits = doc_src.search("mind-bending dreams")
    pretty(hits)

    print("\n== DB ==")
    db_src = DBSource(db_path)
    hits = db_src.search("Inception")
    pretty(hits)

if __name__ == "__main__":
    main()
