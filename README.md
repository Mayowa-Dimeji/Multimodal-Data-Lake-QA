# Unified Data Lake QA with Multi‑Modal Retrieval (RAG over Data Lakes)

A portfolio‑grade project that demonstrates **Retrieval‑Augmented Generation (RAG)** over a **data lake** with **heterogeneous modalities**: relational DB tables, CSV/Excel sheets, and unstructured documents. The system unifies retrieval across sources, normalizes results into a common **Evidence Pack**, and synthesizes **grounded** answers with an LLM—citing the source modality (`[DB]`, `[CSV]`, `[DOC]`) for each claim.

> This project complements a RAG chatbot and a KG+RAG QA system by focusing on **data‑lake RAG** and **fusion of structured + unstructured evidence**—aligned with PhD‑style research in multi‑modal QA.

---

## Table of Contents

- [Goals](#goals)
- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Data Lake Schema (Toy Nolan Dataset)](#data-lake-schema-toy-nolan-dataset)
- [Setup (macOS Apple Silicon + VS Code)](#setup-macos-apple-silicon--vs-code)
- [Quickstart](#quickstart)
- [Step-by-Step Build Guide](#step-by-step-build-guide)
  - [Step 1 — Scaffold + Heterogeneous Data Ingestion](#step-1--scaffold--heterogeneous-data-ingestion)
  - [Step 2 — Unified Retrieval Layer](#step-2--unified-retrieval-layer)
  - [Step 3 — Evidence Fusion + Schema Alignment](#step-3--evidence-fusion--schema-alignment)
  - [Step 4 — LLM Answer Synthesis (Multi‑Modal RAG)](#step-4--llm-answer-synthesis-multi-modal-rag)
  - [Step 5 — Hybrid Query Routing](#step-5--hybrid-query-routing)
  - [Step 6 — Evaluation Harness](#step-6--evaluation-harness)
  - [Step 7 — Productization (API + UI)](#step-7--productization-api--ui)
  - [Step 8 — Advanced Stretch](#step-8--advanced-stretch)
- [Evidence Pack Schema](#evidence-pack-schema)
- [Router Design](#router-design)
- [Prompts & Grounding Rules](#prompts--grounding-rules)
- [API Contracts](#api-contracts)
- [Evaluation Protocol](#evaluation-protocol)
- [UI/UX Notes](#uiux-notes)
- [Configuration & Secrets](#configuration--secrets)
- [Troubleshooting (Apple Silicon)](#troubleshooting-apple-silicon)
- [Roadmap](#roadmap)
- [License](#license)

---

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

## Repository Structure

```
data-lake-rag/
  data_lake/
    csv/                # movies.csv, ratings.csv
    docs/               # *.txt snippets (e.g., interstellar.txt, inception.txt)
    db/                 # seed.sql and movies.db (after seeding)
  etl/
    seed_db.py          # build SQLite from seed.sql
    build_vectors.py    # build embeddings and optional FAISS index for docs
  loaders/
    __init__.py
    common.py           # Evidence dataclass
    csv_loader.py       # CSVSource (fuzzy search via rapidfuzz)
    docs_loader.py      # DocSource (fuzzy; embeddings used by retriever)
    db_loader.py        # DBSource (LIKE query; template SQL in later steps)
  retrievers/
    __init__.py
    structured.py       # wraps CSV + DB
    unstructured.py     # embedding retriever (FAISS or NumPy)
    unified.py          # calls all modalities
  fusion/
    normalize.py        # (Step 3) normalize rows/passages to a common schema
  rag/
    answer.py           # (Step 4) LLM synthesis with grounding rules
    prompts/
      answer_system.md
      cite_instructions.md
  router/
    route.py            # (Step 5) heuristics + optional LLM classifier
  eval/
    benchmark.py        # (Step 6) metrics + reports
    gold/
      queries.jsonl
      labels.jsonl
  api/
    main.py             # (Step 7) FastAPI server
  ui/
    streamlit_app.py    # (Step 7) Streamlit demo (or web/ React app)
  tests/
  .env.example
  README.md
```

> **Note:** Step 1 and Step 2 code is included. Later steps are scaffolded in the plan below.

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

# 2) Install core deps
pip install -r requirements.txt

# 3) (If torch is missing) Install a CPU build that works on M1
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4) Seed the SQLite DB
python etl/seed_db.py

# 5) Build doc embeddings (and FAISS index if faiss-cpu is installed)
python etl/build_vectors.py
```

**Requirements highlights**

- `pandas`, `pyarrow`, `rapidfuzz`
- `numpy`, `sentence-transformers`, `faiss-cpu` (optional but recommended)
- `torch` (for sentence-transformers; CPU build is fine on M1)

---

## Quickstart

```bash
# Smoke test simple per‑source searches (from Step 1)
python demo_search.py

# Unified retrieval across CSV + DB + DOC (from Step 2)
python query_all.py
```

You should see top‑k hits for each modality with basic scores.

---

## Step-by-Step Build Guide

### Step 1 — Scaffold + Heterogeneous Data Ingestion

**Goal:** set up a minimal data lake and per‑source loaders.

- Data in `data_lake/` (CSV, docs, DB via `seed_db.py`).
- Loaders in `loaders/` with a shared `Evidence` dataclass.
- Smoke test: `python demo_search.py`.

### Step 2 — Unified Retrieval Layer

**Goal:** query all sources with a single call.

- Structured: `retrievers/structured.py` → CSV/DB top‑k.
- Unstructured: `retrievers/unstructured.py` → embeddings + FAISS/NumPy.
- Unified: `retrievers/unified.py` → top‑k per modality.
- Build vectors: `python etl/build_vectors.py`.
- Smoke test: `python query_all.py`.

### Step 3 — Evidence Fusion + Schema Alignment

**Goal:** produce a **consistent Evidence Pack** for the LLM.

- Implement `fusion/normalize.py` to convert:
  - DB/CSV rows → **compact tables** + **semantic triples**.
  - DOC passages → **text chunks** with metadata (doc, offsets).
- Output JSON (`/tmp/evidence.json` or return from function):
  ```json
  {
    "query": "...",
    "retrieval": {
      "db": [ { "source_id": "...", "row": {...}, "score": 0.97 } ],
      "csv": [ { "source_id": "...", "row": {...}, "score": 0.94 } ],
      "docs":[ { "source_id": "...", "chunk": "...", "score": 0.89 } ]
    },
    "entities": { "canonical_map": {"INCEPTION (2010)": "Inception"} }
  }
  ```

### Step 4 — LLM Answer Synthesis (Multi‑Modal RAG)

**Goal:** grounded answers with modality citations.

- Implement `rag/answer.py` with strict prompts:
  - Cite each claim with `[DB]`, `[CSV]`, or `[DOC]`.
  - “Answer only from evidence. If insufficient, say so.”
- Parse output to JSON: `{ "answer": "...", "citations": [ ... ], "used_modalities": ["DB","CSV"] }`.

### Step 5 — Hybrid Query Routing

**Goal:** choose structured/unstructured/both.

- `router/route.py`:
  - Heuristics: numbers/dates → Structured; opinions/themes → Unstructured; comparisons → Both.
  - Optional LLM backstop → `{"route":"structured|unstructured|both","confidence":0.0-1.0}`.
- Log routing decisions for later evaluation.

### Step 6 — Evaluation Harness

**Goal:** measure retrieval + routing + attribution + answer quality.

- `eval/benchmark.py`:
  - Structured P@k/R@k (did we retrieve the correct row?).
  - Unstructured nDCG@k.
  - Routing accuracy (vs. gold labels).
  - Attribution correctness (% claims citing allowed sources).
  - Answer quality: exact/approx string match or LLM‑as‑judge rubric.
- Datasets in `eval/gold/` (50–100 queries).

### Step 7 — Productization (API + UI)

**Goal:** demo‑ready app.

- API: `api/main.py` (FastAPI) endpoints:
  - `POST /query` → returns `{answer, citations, evidence_pack}`.
  - `POST /reroute` → override route for ablations.
- UI: `ui/streamlit_app.py` or React app:
  - Query box, router badge, inline citations.
  - Evidence tabs for DB/CSV/DOC.
  - “Show Prompt” and “Evidence JSON” toggles.

### Step 8 — Advanced Stretch

- Cross‑modal joins (e.g., DB titles ↔ DOC mentions of “mind‑bending”).
- Data freshness: periodic sync from OMDb (or other APIs).
- Multi‑hop reasoning across modalities.

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

**LLM backstop (optional):**

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

### REST (FastAPI, Step 7)

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

## Configuration & Secrets

Create `.env` from `.env.example`:

```
# LLM provider (Step 4+)
OPENAI_API_KEY=
# Optional vector DB settings (if switching from FAISS to Qdrant, etc.)
QDRANT_URL=
QDRANT_API_KEY=
```

---

## Troubleshooting (Apple Silicon)

- **Torch install**: use the CPU wheels:  
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **faiss-cpu**: if a prebuilt wheel fails, try:  
  `pip install faiss-cpu` (works on most M1 setups). If not, rely on NumPy fallback.
- **SentenceTransformer download timeouts**: ensure stable network; try a different Python mirror if needed.
- **SQLite file locks**: avoid parallel writers; prefer read‑only during retrieval.

---

## Roadmap

- Add **chunking with overlap** for longer docs; store offsets and render “open in context” in UI.
- Introduce **DuckDB/Parquet** pathway for larger CSVs; vector‑append flow for new docs.
- Implement **LLM‑backed join inference** between structured rows and doc mentions.
- Add **freshness sync** from a public API (e.g., OMDb) and a scheduler.
- Expand the **gold set** to 100–200 queries with richer comparisons.

---

## License

**MIT** — feel free to use and adapt this project for personal, academic, or commercial demos. Please keep a reference to this repository structure/design if it helps your readers/users.
