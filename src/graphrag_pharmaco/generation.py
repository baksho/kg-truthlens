"""Grounded answer generation: structured JSON output with citations and an explicit abstain flag."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from graphrag_pharmaco.llm import LLM
from graphrag_pharmaco.retrieval.base import Retrieval

SYSTEM = """You answer questions about a pharmacogenomics knowledge graph using ONLY the numbered
evidence provided. Reply with a single JSON object and nothing else:
{"answer": [<strings>], "citations": [<evidence ids>], "abstain": <true|false>, "note": <string>}

Rules:
- Every item in "answer" must be an entity label (or value) that appears in the cited evidence,
  written exactly as it appears there.
- "citations" lists the ids (E1, E2, ...) that support the answer. Cite only ids that exist.
- Evidence marked PREDICTED is model output, not measurement. If you use it, say so in "note" and
  never present it as measured. MEASURED evidence is experimental data.
- If the evidence does not contain the answer, set "abstain" to true and "answer" to []. Do not
  use outside knowledge. Do not guess.
"""


@dataclass
class Answer:
    answer: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    abstain: bool = False
    note: str = ""
    parse_ok: bool = True
    raw: str = ""


def build_prompt(question: str, retrieval: Retrieval) -> str:
    if retrieval.evidence:
        evidence = "\n".join(f"[{e.id}] {e.text}" for e in retrieval.evidence)
    else:
        evidence = "(no evidence retrieved)"
    return f"Question: {question}\n\nEvidence:\n{evidence}"


def parse_answer(raw: str) -> Answer:
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return Answer(parse_ok=False, raw=raw)
    try:
        obj = json.loads(raw[start : end + 1])
        answer = [str(x) for x in obj.get("answer", [])]
        citations = [str(x) for x in obj.get("citations", [])]
        return Answer(answer, citations, bool(obj.get("abstain", False)), str(obj.get("note", "")), True, raw)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return Answer(parse_ok=False, raw=raw)


def generate(question: str, retrieval: Retrieval, llm: LLM) -> Answer:
    if not retrieval.evidence:
        # Nothing retrieved: abstain without spending an LLM call or inviting a guess.
        return Answer(abstain=True, note="no evidence retrieved")
    return parse_answer(llm.complete(SYSTEM, build_prompt(question, retrieval)))
