
from __future__ import annotations
import os, json, time, argparse, sys
from pathlib import Path

# Ensure repo root on sys.path when executed directly
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from typing import List
from retrievers import UnifiedRetriever, StructuredRetriever, UnstructuredRetriever
from fusion import normalize_retrieval
from router.route import route_query

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=False, default="Which Nolan movie has the highest IMDb rating?")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--route", type=str, default="auto", choices=["auto","structured","unstructured","both"], help="Force a route or auto-route.")
    p.add_argument("--use-llm-router", action="store_true", help="Use LLM backstop for routing (requires OPENAI_API_KEY).")
    p.add_argument("--use-llm", action="store_true", help="Use LLM for answer synthesis (requires OPENAI_API_KEY).")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    args = p.parse_args()

    csv_paths = [
        BASE / "data_lake" / "csv" / "movies.csv",
        BASE / "data_lake" / "csv" / "ratings.csv",
    ]
    db_path = BASE / "data_lake" / "db" / "movies.db"
    docs_index = BASE / "indexes" / "docs"

    # Decide route
    if args.route == "auto":
        route, conf, feats = route_query(args.query, use_llm=args.use_llm_router, model=args.model)
    else:
        route, conf, feats = (args.route, 1.0, {"forced": True})

    # Retrieve per route
    if route == "structured":
        from retrievers import StructuredRetriever
        retr = StructuredRetriever(csv_paths=csv_paths, db_path=db_path)
        struct = retr.search(args.query, k_per_modality=args.k)
        retrieval_dict = {
            "db": [{"origin":"DB","source_id":h.source_id,"score":float(h.score),"payload":h.payload} for h in struct.get("db",[])],
            "csv": [{"origin":"CSV","source_id":h.source_id,"score":float(h.score),"payload":h.payload} for h in struct.get("csv",[])],
            "docs": []
        }
    elif route == "unstructured":
        from retrievers import UnstructuredRetriever
        retr = UnstructuredRetriever(index_dir=docs_index)
        docs = retr.search(args.query, k=args.k)
        retrieval_dict = {"db": [], "csv": [], "docs": [{"origin":"DOC","source_id":h.source_id,"score":float(h.score),"payload":h.payload} for h in docs]}
    else:  # both
        retr = UnifiedRetriever(csv_paths=csv_paths, db_path=db_path, docs_index_dir=docs_index)
        out = retr.search_all(args.query, k_per_modality=args.k)
        def ser(hits):
            return [{"origin":h.origin, "source_id":h.source_id, "score":float(h.score), "payload":h.payload} for h in hits]
        retrieval_dict = {"db": ser(out["db"]), "csv": ser(out["csv"]), "docs": ser(out["docs"])}

    pack = normalize_retrieval(query=args.query, retrieval=retrieval_dict)

    from rag.answer import synthesize_answer
    answer = synthesize_answer(pack, prefer_llm=args.use_llm, model=args.model)

    # Save
    outputs = BASE / "outputs"
    outputs.mkdir(exist_ok=True, parents=True)
    ts = int(time.time())
    outpath = outputs / f"answer_{ts}.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump({"query": args.query, "route": route, "route_confidence": conf, "answer": answer, "evidence": pack}, f, ensure_ascii=False, indent=2)

    print(f"\nRoute: {route} (conf={conf:.2f})  Query: {args.query}\n")
    print("Answer:\n" + answer.get("answer","(no answer)"))
    print(f"\nUsed modalities: {', '.join(answer.get('used_modalities', [])) or '(none)'}")
    print(f"\nSaved → {outpath}")

if __name__ == "__main__":
    main()
