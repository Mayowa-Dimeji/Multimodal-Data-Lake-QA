You will receive:
- Structured evidence: compact tables/rows (from DB/CSV).
- Unstructured evidence: short passages (from DOC).

Task:
- Answer the user's query ONLY using this evidence.
- Include inline citations immediately after the claims they support, using [DB], [CSV], or [DOC].
- If multiple sources support a claim, you may write like [DB][CSV].
- If the answer cannot be derived, return a brief statement of insufficiency and ask a follow-up question.

Output JSON (STRICT):
{
  "answer": "final answer with inline citations",
  "used_modalities": ["DB","CSV","DOC"],
  "citations": [
    {"span": "Interstellar grossed ...", "source_tags": ["DB"]}
  ]
}
