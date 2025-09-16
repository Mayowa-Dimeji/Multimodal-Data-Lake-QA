
from __future__ import annotations
import os, json, re, time
from pathlib import Path
from typing import Dict, Any, List

# Optional deps
try:
    from dotenv import load_dotenv # type: ignore
    load_dotenv()
except Exception:
    pass

def _format_structured(evidence: Dict[str, Any]) -> str:
    """Render DB/CSV rows as compact tables in markdown-like format."""
    lines: List[str] = []
    # DB
    for h in evidence.get("db", [])[:5]:
        row = h.get("row", {})
        subset = {k: row.get(k) for k in ("title","release_year","box_office_usd","runtime_min","imdb","metacritic") if k in row}
        lines.append(f"- [DB] {subset}")
    # CSV
    for h in evidence.get("csv", [])[:5]:
        row = h.get("row", {})
        subset = {k: row.get(k) for k in ("title","release_year","imdb","metacritic","rt_tomatoes") if k in row}
        lines.append(f"- [CSV] {subset}")
    return "\n".join(lines) if lines else "(none)"

def _format_unstructured(evidence: Dict[str, Any]) -> str:
    lines: List[str] = []
    for h in evidence.get("docs", [])[:5]:
        chunk = h.get("chunk","").replace("\n"," ")
        doc = h.get("metadata",{}).get("doc","")
        lines.append(f"- [DOC] ({doc}) {chunk[:400]}")
    return "\n".join(lines) if lines else "(none)"

def build_prompt(pack: Dict[str, Any]) -> Dict[str, str]:
    """Return dict with 'system' and 'user' strings."""
    query = pack.get("query","")
    struct = _format_structured(pack.get("retrieval",{}))
    unstruct = _format_unstructured(pack.get("retrieval",{}))

    system_path = Path(__file__).resolve().parent / "prompts" / "answer_system.md"
    cite_path = Path(__file__).resolve().parent / "prompts" / "cite_instructions.md"
    system_rules = system_path.read_text()
    cite_rules = cite_path.read_text()

    user = f"""# Question
{query}

# Structured Evidence (tables/rows)
{struct}

# Unstructured Evidence (passages)
{unstruct}

# Instructions
{cite_rules}
"""
    return {"system": system_rules, "user": user}

def _extract_json(text: str) -> Dict[str, Any]:
    """Try to extract a JSON object from a model response or return empty dict."""
    # Look for fenced blocks
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}

def _fallback_compose(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Compose a minimal grounded answer without calling an LLM."""
    db = pack.get("retrieval",{}).get("db",[])
    csv = pack.get("retrieval",{}).get("csv",[])
    docs = pack.get("retrieval",{}).get("docs",[])

    lines: List[str] = []
    used = set()

    # Try to find top DB facts
    for h in db[:2]:
        row = h.get("row",{})
        title = row.get("title")
        year = row.get("release_year")
        boxo = row.get("box_office_usd")
        if title and year:
            if boxo:
                lines.append(f"- {title} ({year}) grossed ${boxo:,} [DB].")
            else:
                lines.append(f"- {title} ({year}) [DB].")
            used.add("DB")

    # Ratings from CSV
    for h in csv[:2]:
        row = h.get("row",{})
        title = row.get("title")
        imdb = row.get("imdb")
        meta = row.get("metacritic")
        bits = []
        if imdb is not None:
            bits.append(f"IMDb {imdb}")
        if meta is not None:
            bits.append(f"Metacritic {meta}")
        if title and bits:
            lines.append(f"- {title} ratings: {', '.join(bits)} [CSV].")
            used.add("CSV")

    # Themes from docs
    if docs:
        chunk = docs[0].get("chunk","").strip().replace("\n"," ")
        if chunk:
            lines.append(f"- Critics note: {chunk[:200]} [DOC].")
            used.add("DOC")

    if not lines:
        lines.append("Insufficient evidence in the pack to answer. Please refine the query.")
    return {"answer": "\n".join(lines), "used_modalities": sorted(list(used)), "citations": []}

def synthesize_with_openai(prompt: Dict[str,str], model: str = "gpt-4o-mini", timeout: int = 60) -> Dict[str, Any]:
    """Use OpenAI if available; otherwise raise to trigger fallback."""
    try:
        import openai # type: ignore
    except Exception as e:
        raise RuntimeError("openai package not installed") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = api_key

    # Use Chat Completions (SDK v0.28 style) or new client if v1+ is installed.
    try:
        from openai import OpenAI # type: ignore
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":prompt["system"]},
                {"role":"user","content":prompt["user"]},
            ],
            temperature=0.0,
            max_tokens=600,
            timeout=timeout,
        )
        text = resp.choices[0].message.content
    except Exception:
        # Fallback to legacy
        text = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role":"system","content":prompt["system"]},
                {"role":"user","content":prompt["user"]},
            ],
            temperature=0.0,
            max_tokens=600,
            request_timeout=timeout,
        )["choices"][0]["message"]["content"]

    data = _extract_json(text)
    if not data:
        # If LLM didn't return JSON, wrap as best-effort
        data = {"answer": text.strip(), "used_modalities": [], "citations": []}
    return data

def synthesize_answer(pack: Dict[str, Any], prefer_llm: bool = True, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    prompt = build_prompt(pack)
    if prefer_llm:
        try:
            return synthesize_with_openai(prompt, model=model)
        except Exception:
            pass
    # Fallback deterministic composition
    return _fallback_compose(pack)
