
from __future__ import annotations
import os, json, time, argparse
from pathlib import Path
from typing import List
from retrievers import UnifiedRetriever
from fusion import normalize_retrieval
from rag.answer import synthesize_answer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=False, default="Which Nolan movie has the highest IMDb rating?")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--use-llm", action="store_true", help="Use OpenAI if available; otherwise fallback to deterministic answer.")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    args = p.parse_args()

    BASE = Path(__file__).resolve().parent.parent
    csv_paths = [
        BASE / "data_lake" / "csv" / "movies.csv",
        BASE / "data_lake" / "csv" / "ratings.csv",
    ]
    db_path = BASE / "data_lake" / "db" / "movies.db"
    docs_index = BASE / "indexes" / "docs"

    retr = UnifiedRetriever(csv_paths=csv_paths, db_path=db_path, docs_index_dir=docs_index)
    out = retr.search_all(args.query, k_per_modality=args.k)

    # Convert Evidence objects to dicts
    def ser(hits):
        return [{"origin":h.origin, "source_id":h.source_id, "score":float(h.score), "payload":h.payload} for h in hits]

    retrieval_dict = {"db": ser(out["db"]), "csv": ser(out["csv"]), "docs": ser(out["docs"])}
    pack = normalize_retrieval(query=args.query, retrieval=retrieval_dict)

    answer = synthesize_answer(pack, prefer_llm=args.use_llm, model=args.model)

    # Save
    outputs = BASE / "outputs"
    outputs.mkdir(exist_ok=True, parents=True)
    ts = int(time.time())
    outpath = outputs / f"answer_{ts}.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump({"query": args.query, "answer": answer, "evidence": pack}, f, ensure_ascii=False, indent=2)

    print(f"\nQuery: {args.query}\n")
    print("Answer:\n" + answer.get("answer","(no answer)"))
    print(f"\nUsed modalities: {', '.join(answer.get('used_modalities', [])) or '(none)'}")
    print(f"\nSaved → {outpath}")

if __name__ == "__main__":
    main()
