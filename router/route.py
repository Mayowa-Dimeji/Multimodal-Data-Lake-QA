
from __future__ import annotations
from typing import Dict, Tuple, Optional
import os

# Optional LLM backstop
def _llm_route(query: str, model: str = "gpt-4o-mini") -> Optional[Tuple[str, float]]:
    """
    Ask an LLM to pick a route. Returns (route, confidence) or None if unavailable.
    """
    try:
        import openai  # type: ignore # legacy import path fallback
    except Exception:
        openai = None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or openai is None:
        return None

    try:
        from openai import OpenAI # type: ignore
        client = OpenAI(api_key=api_key)
        prompt = (
            "Route the user query to one of: structured, unstructured, both.\n"
            "structured: numeric/date facts, counts, filters, aggregates, exact release years.\n"
            "unstructured: opinions, themes, sentiment, long-form descriptions.\n"
            "both: comparisons across multiple entities mixing facts and descriptions.\n"
            f"Query: {query}\n"
            "Respond as JSON: {\"route\":\"structured|unstructured|both\",\"confidence\":0.0-1.0}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        text = resp.choices[0].message.content.strip()
    except Exception:
        # Legacy SDK
        text = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            max_tokens=60,
        )["choices"][0]["message"]["content"].strip()
    import json, re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        route = data.get("route")
        conf = float(data.get("confidence", 0.0))
        if route in ("structured","unstructured","both"):
            return (route, conf)
    except Exception:
        return None
    return None

STRUCTURED_CUES = set("""max highest lowest average sum count how many total runtime budget box office revenue year released release_year imdb metacritic rating rt tomato score numeric number greater less before after since between top compare vs vs. difference""".split())
UNSTRUCTURED_CUES = set("""theme themes critics say review describe described described as plot summary opinion sentiment tone character relationship emotional""".split())
COMPARATIVE_CUES = set("""compare vs versus both and contrast than between against""".split())

def heuristic_route(query: str) -> Tuple[str, float, Dict[str, bool]]:
    """
    Return (route, confidence, features) based on simple token cues.
    """
    q = (query or "").lower()
    tokens = set(q.replace("?"," ").replace(","," ").split())

    has_struct = any(cue in q for cue in STRUCTURED_CUES)
    has_unstruct = any(cue in q for cue in UNSTRUCTURED_CUES)
    has_compare = any(cue in q for cue in COMPARATIVE_CUES) or (" and " in q and (" vs " in q or " compare " in q))

    if has_compare and (has_struct or has_unstruct):
        return ("both", 0.85, {"structured": has_struct, "unstructured": has_unstruct, "comparative": True})
    if has_struct and not has_unstruct:
        return ("structured", 0.8, {"structured": True, "unstructured": False, "comparative": has_compare})
    if has_unstruct and not has_struct:
        return ("unstructured", 0.8, {"structured": False, "unstructured": True, "comparative": has_compare})
    # Fallback: if asking for "which/when/how many" -> structured; if "why/describe" -> unstructured
    if any(k in q.split() for k in ("which","when","how","many","list","show")):
        return ("structured", 0.6, {"structured": True, "unstructured": False, "comparative": has_compare})
    if any(k in q.split() for k in ("why","describe","explain","theme","themes")):
        return ("unstructured", 0.6, {"structured": False, "unstructured": True, "comparative": has_compare})
    # Default
    return ("both", 0.5, {"structured": has_struct, "unstructured": has_unstruct, "comparative": has_compare})

def route_query(query: str, use_llm: bool = False, model: str = "gpt-4o-mini") -> Tuple[str, float, Dict[str,bool]]:
    """
    Decide route: 'structured' | 'unstructured' | 'both'.
    If use_llm is True and OPENAI is configured, will backstop the heuristic.
    """
    r, conf, feats = heuristic_route(query)
    if use_llm:
        llm = _llm_route(query, model=model)
        if llm is not None:
            lr, lc = llm
            # Blend decisions: prefer llm if confident, else heuristic
            if lc >= conf or (lr != r and lc >= 0.7):
                return (lr, lc, feats)
    return (r, conf, feats)
