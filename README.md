# Data Lake RAG — Step 1: Scaffold + Heterogeneous Data Ingestion

This starter contains:
- `data_lake/csv`: `movies.csv`, `ratings.csv`
- `data_lake/docs`: 2 small review snippets
- `data_lake/db`: `seed.sql` + script to build `movies.db`
- `loaders/`: CSV, DOC, and SQLite loaders with a common Evidence dataclass
- `demo_search.py`: quick smoke test

## Quickstart (macOS Apple Silicon)

```bash
# 1) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Build the SQLite database
python etl/seed_db.py

# 4) Run a smoke test across all sources
python demo_search.py
```

If everything works, you should see hits from CSV, DOC, and DB sources.

In Step 2 we'll add a proper unified retrieval layer, FAISS/Qdrant (optional), and per-modality k-best APIs.
