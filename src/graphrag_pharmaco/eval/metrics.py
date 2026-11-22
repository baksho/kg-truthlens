"""Metrics. Faithfulness is checked on structured output, not on free-text claim extraction.

- precision/recall/F1: predicted answer set vs gold set (answerable questions).
- abstain_correct: on unanswerable questions, did the system abstain (or return nothing)?
- over_abstain: on answerable questions, did it abstain anyway?
- citation_validity: share of cited ids that exist in the retrieved evidence.
- unsupported_rate: share of answer items that do not occur in the retrieved evidence text.
- retrieval_recall: share of gold items present in the evidence (no LLM needed).
- linking_recall: share of expected entities the linker found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from graphrag_pharmaco.eval.questions import Question
from graphrag_pharmaco.generation import Answer
from graphrag_pharmaco.retrieval.base import Retrieval


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


@dataclass
class ItemScore:
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    abstain_correct: float | None = None
    over_abstain: float | None = None
    citation_validity: float | None = None
    unsupported_rate: float | None = None
    retrieval_recall: float | None = None
    linking_recall: float | None = None
    sparql_valid: float | None = None
    parse_ok: float | None = None


def score_retrieval(q: Question, r: Retrieval) -> dict[str, float | None]:
    haystack = norm(" \n ".join(e.text for e in r.evidence))
    rec = None
    if q.answerable and q.gold:
        rec = sum(norm(g) in haystack for g in q.gold) / len(q.gold)
    link = None
    if q.entities:
        link = sum(u in set(r.linked) for u in q.entities) / len(q.entities) if r.linked or r.strategy in (
            "neighborhood", "hybrid") else None
    return {
        "retrieval_recall": rec,
        "linking_recall": link,
        "sparql_valid": None if r.sparql_valid is None else float(r.sparql_valid),
    }


def score_answer(q: Question, r: Retrieval, a: Answer) -> ItemScore:
    s = ItemScore(**score_retrieval(q, r))
    s.parse_ok = float(a.parse_ok)
    abstained = a.abstain or (a.parse_ok and not a.answer)
    if q.answerable:
        gold, pred = {norm(x) for x in q.gold}, {norm(x) for x in a.answer}
        tp = len(gold & pred)
        s.precision = tp / len(pred) if pred else 0.0
        s.recall = tp / len(gold) if gold else 0.0
        s.f1 = 0.0 if tp == 0 else 2 * s.precision * s.recall / (s.precision + s.recall)
        s.over_abstain = float(bool(abstained))
    else:
        s.abstain_correct = float(bool(abstained))
    if a.citations:
        ids = {e.id for e in r.evidence}
        s.citation_validity = sum(c in ids for c in a.citations) / len(a.citations)
    if a.answer:
        haystack = norm(" \n ".join(e.text for e in r.evidence))
        s.unsupported_rate = sum(norm(x) not in haystack for x in a.answer) / len(a.answer)
    return s


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
