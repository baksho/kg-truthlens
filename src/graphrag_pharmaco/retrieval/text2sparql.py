"""Strategy: the LLM writes SPARQL from the schema; we validate, run and retry on errors."""

from __future__ import annotations

from graphrag_pharmaco.kg.backend import SparqlBackend
from graphrag_pharmaco.kg.guard import QueryRejected, extract_sparql, guard
from graphrag_pharmaco.kg.labels import LabelStore
from graphrag_pharmaco.kg.linking import LinkedEntity
from graphrag_pharmaco.kg.schema import Schema
from graphrag_pharmaco.llm import LLM
from graphrag_pharmaco.retrieval.base import Retrieval, Timer, number

SYSTEM = """You translate questions into ONE SPARQL 1.1 SELECT query for the schema below.
Reply with only the query in a ```sparql fenced block. No explanation.

{schema}
Rules:
- Match entities by label when no URI is given: ?e rdfs:label ?l . FILTER(LCASE(STR(?l)) = "name")
- Use ph:Response for measured data and ph:PredictedResponse only when the question asks about
  predictions. Never mix them in one result.
- Select human-readable labels (rdfs:label), not bare URIs, in the result columns.
- If the schema cannot answer the question, still write the closest valid query; do not invent
  properties.

Example 1
Question: Which genes does DrugX target?
```sparql
SELECT ?answer WHERE {{ ?d rdfs:label ?dl . FILTER(LCASE(STR(?dl)) = "drugx")
  ?d ph:targets ?g . ?g rdfs:label ?answer }}
```
Example 2
Question: Which 3 cell lines are most sensitive to DrugX?
```sparql
SELECT ?answer WHERE {{ ?d rdfs:label ?dl . FILTER(LCASE(STR(?dl)) = "drugx")
  ?r a ph:Response ; ph:ofDrug ?d ; ph:ofCellLine ?c ; ph:auc ?auc . ?c rdfs:label ?answer }}
ORDER BY ?auc LIMIT 3
```
"""


class Text2SparqlRetriever:
    name = "text2sparql"

    def __init__(
        self,
        backend: SparqlBackend,
        schema: Schema,
        labels: LabelStore,
        llm: LLM,
        max_rows: int = 200,
        max_attempts: int = 3,
    ):
        self.backend = backend
        self.schema = schema
        self.labels = labels
        self.llm = llm
        self.max_rows = max_rows
        self.max_attempts = max_attempts

    def retrieve(self, question: str) -> Retrieval:
        return self.retrieve_with_hints(question, [])

    def retrieve_with_hints(
        self, question: str, hints: list[LinkedEntity], strategy: str | None = None
    ) -> Retrieval:
        system = SYSTEM.format(schema=self.schema.description())
        user = f"Question: {question}"
        if hints:
            lines = "\n".join(f"- <{h.uri}>  ({h.cls} \"{h.label}\")" for h in hints)
            user += f"\n\nEntities found in the question (use these URIs exactly):\n{lines}"
        result = Retrieval(strategy or self.name, linked=[h.uri for h in hints], sparql_valid=False)
        with Timer() as t:
            feedback = ""
            for _ in range(self.max_attempts):
                raw = self.llm.complete(system, user + feedback)
                query = self.schema.with_prefixes(extract_sparql(raw))
                result.sparql = query
                try:
                    safe = guard(query, self.max_rows)
                    rows = self.backend.select(safe)
                except QueryRejected as exc:
                    result.error = str(exc)
                    feedback = f"\n\nYour previous query was rejected: {exc}\nFix it."
                    continue
                except Exception as exc:  # backend error (timeout, 400, ...)
                    result.error = f"{type(exc).__name__}: {exc}"
                    feedback = f"\n\nYour previous query failed to execute: {exc}\nFix it."
                    continue
                result.sparql_valid = True
                result.error = None
                status = "predicted" if "predictedresponse" in query.lower() else "measured"
                self.labels.prefetch([v for row in rows for v in row.values()])
                items = [
                    (status, "; ".join(f"{k}: {self.labels.render(v)}" for k, v in row.items()))
                    for row in rows
                ]
                result.evidence = number(items)
                break
        result.latency_s = t.elapsed
        return result
