# Unified Data Lake QA with Multi‑Modal Retrieval (RAG over Data Lakes)

A portfolio‑grade project that demonstrates **Retrieval‑Augmented Generation (RAG)** over a **data lake** with **heterogeneous modalities**: relational DB tables, CSV/Excel sheets, and unstructured documents. The system unifies retrieval across sources, normalizes results into a common **Evidence Pack**, and synthesizes **grounded** answers with an LLM—citing the source modality (`[DB]`, `[CSV]`, `[DOC]`) for each claim.

## Goals

- Build a QA system that **queries across a data lake** of mixed formats (DB, CSV, DOC).
- **Unify retrieval** and **normalize evidence** so an LLM can reason across modalities.
- Enforce **grounding and attribution** with per‑claim citations and modality tags.
- Provide a **reproducible evaluation harness** to compare **text‑only RAG vs. data‑lake RAG**.

## Features

- Heterogeneous ingestion: CSV/Excel, SQLite/Postgres, text snippets (docs).
- Unified retrieval with **structured** (SQL/Pandas) and **unstructured** (embeddings + FAISS/NumPy) backends.
- Evidence fusion + schema alignment → a single **Evidence Pack** JSON.
- LLM synthesis with **strict grounding** and **modality citations** (`[DB]`, `[CSV]`, `[DOC]`).
- Hybrid query **router** (heuristics + optional LLM classifier).
- **Evaluation harness** (retrieval metrics, routing accuracy, attribution correctness).
- Demo **API (FastAPI)** and **UI (Streamlit/React)** with evidence tabs and inline citations.

---

## Architecture

```
            ┌──────────────────┐
            │   Data Lake      │
            │  CSV | DB | DOC  │
            └────────┬─────────┘
                     │ ingestion (ETL)
                     ▼
             ┌───────────────┐
             │  Retrievers   │
             │  Structured   │───► CSV & DB hits
             │  Unstructured │───► DOC passages
             └───────┬───────┘
                     │
                     ▼
            ┌───────────────────┐
            │  Fusion/Normalize │  → Evidence Pack (JSON)
            └────────┬──────────┘
                     │
                     ▼
             ┌───────────────┐
             │    Router     │  (Structured | Unstructured | Both)
             └────────┬──────┘
                     │
                     ▼
             ┌───────────────┐
             │ LLM Answerer  │  → Grounded answer + citations
             └───────────────┘
                     │
                     ▼
             ┌───────────────┐
             │ API &  UI     │
             └───────────────┘
```

---

## Data Lake Schema (Toy Nolan Dataset)

- **DB (SQLite) — `movies` table**: `title`, `release_year`, `director`, `box_office_usd`, `runtime_min`, `genres`.
- **CSV — `movies.csv`, `ratings.csv`**: metadata & critic scores (IMDB, Metacritic, RottenTomatoes).
- **Docs — `*.txt`**: short critic‑style snippets (e.g., Interstellar themes).

---

## Setup (macOS Apple Silicon + VS Code)

```bash
# 1) Create a Python venv
python3 -m venv .venv
source .venv/bin/activate


pip install -r requirements.txt


pip install torch --index-url https://download.pytorch.org/whl/cpu

# Seed the SQLite DB
python etl/seed_db.py

# 5) Build doc embeddings
python etl/build_vectors.py
```

**Requirements highlights**

- `pandas`, `pyarrow`, `rapidfuzz`
- `numpy`, `sentence-transformers`, `faiss-cpu`
- `torch`

---

## Evidence Pack Schema

```json
{
  "query": "Compare Inception and Interstellar’s box office and themes.",
  "retrieval": {
    "db": [
      {
        "source_id": "db:movies:inception",
        "table": "movies",
        "row": {
          "title": "Inception",
          "release_year": 2010,
          "box_office_usd": 829895144
        },
        "score": 0.97
      }
    ],
    "csv": [
      {
        "source_id": "csv:ratings:interstellar",
        "file": "ratings.csv",
        "row": { "title": "Interstellar", "imdb": 8.7 },
        "score": 0.94
      }
    ],
    "docs": [
      {
        "source_id": "doc:reviews:interstellar_42",
        "chunk": "Critics describe Interstellar as a meditation on love and time...",
        "metadata": {
          "doc": "interstellar_reviews.txt",
          "start": 1200,
          "end": 1600
        },
        "score": 0.89
      }
    ]
  },
  "entities": {
    "canonical_map": { "INCEPTION (2010)": "Inception" }
  }
}
```

---

## Router Design

**Heuristics (transparent, fast):**

- Numbers/dates/aggregations (`max`, `sum`, `count`, `highest`, `released`) → **Structured**.
- Opinions/themes/sentiment/summaries (`critics say`, `themes`, `tone`) → **Unstructured**.
- Comparative/multi‑entity (`compare`, `vs`, `both`, `and`) → **Both**.

**LLM backstop :**

- Short “route‑only” prompt that returns JSON with `route` and `confidence`.
- Use only when heuristics are ambiguous; log all decisions.

---

## Prompts & Grounding Rules

- **System Prompt (core):**
  - “You must answer **only** from the supplied evidence. Cite each claim with `[DB]`, `[CSV]`, or `[DOC]`. If evidence is insufficient, state that explicitly and request a clarifying query.”
- **Answer formatting:**
  - Prefer concise bullets with inline citations:  
    `Interstellar grossed ~$677M [DB]. Critics describe it as a meditation on love and time [DOC].`
- **Refuse hallucinations:** any unsupported claim must be omitted or flagged.

---

## API Contracts

### Retriever (Python)

```python
class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> List[Evidence]: ...
```

### Unified Retriever (Python)

```python
def search_all(query: str, k_per_modality: int = 5) -> Dict[str, List[Evidence]]
```

### REST

```
POST /query
{
  "query": "Which Nolan movie had the highest IMDb rating?"
}
→ 200
{
  "answer": "...",
  "citations": [...],
  "route": "structured|unstructured|both",
  "evidence": {...}  # Evidence Pack
}
```

---

## Evaluation Protocol

- **Structured retrieval:** P@k, R@k on labeled target rows/keys.
- **Unstructured retrieval:** nDCG@k with graded relevance.
- **Routing accuracy:** exact match to gold modality labels.
- **Attribution correctness:** % claims citing allowed sources for that query.
- **Answer quality:** exact/approx matching for factual; rubric/LLM for explanatory.

Baselines: compare **text‑only RAG** vs. **data‑lake RAG** on the same gold set.

---

## UI/UX Notes

- Router badge: `Structured`, `Unstructured`, or `Both`.
- Evidence tabs: 🗄️ DB (table preview), 📊 CSV (dataframe chips), 📄 DOC (highlighted passages).
- Controls: “Re‑route as …”, “Show Prompt”, “Show Evidence JSON” for transparency.
- Error affordances: show “insufficient evidence” with a prompt to refine the query.

---

#
