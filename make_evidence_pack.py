
import json, time
from pathlib import Path
from fusion import normalize_retrieval

BASE = Path(__file__).resolve().parent

def main():
    query = "Compare Inception and Interstellar: box office, IMDb rating, and themes."
    retrieval = {
        "db": [{
            "origin": "DB",
            "source_id": "db:movies:Inception",
            "score": 0.99,
            "payload": {"title": "Inception", "release_year": 2010, "box_office_usd": 829895144}
        }],
        "csv": [{
            "origin": "CSV",
            "source_id": "csv:ratings:interstellar",
            "score": 0.95,
            "payload": {"title": "Interstellar", "imdb": 8.7, "metacritic": 74}
        }],
        "docs": [{
            "origin": "DOC",
            "source_id": "doc:interstellar.txt",
            "score": 0.88,
            "payload": {"doc": "interstellar.txt", "snippet": "Critics describe Interstellar as a meditation on love and time."}
        }]
    }
    pack = normalize_retrieval(query=query, retrieval=retrieval)
    out_dir = BASE / "evidence_packs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"evidence_pack_demo.json"
    path.write_text(json.dumps(pack, indent=2))
    print(f"Wrote demo Evidence Pack → {path}")

if __name__ == "__main__":
    main()
