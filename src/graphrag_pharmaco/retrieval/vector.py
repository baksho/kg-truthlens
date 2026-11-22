"""Baseline strategy: vector search over per-entity text documents (classic RAG).

Each entity is verbalised into one short document built from the KG, then retrieved by TF-IDF
cosine similarity. The embedder is a plain TF-IDF so the baseline needs no model download; swap
in a dense embedder by subclassing and overriding `_encode`.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from graphrag_pharmaco.kg.backend import SparqlBackend
from graphrag_pharmaco.kg.labels import LabelStore
from graphrag_pharmaco.kg.linking import EntityLinker
from graphrag_pharmaco.kg.schema import RDF_TYPE, RDFS_LABEL, Schema
from graphrag_pharmaco.retrieval.base import Retrieval, Timer, number


class VectorRetriever:
    name = "vector"

    def __init__(
        self,
        backend: SparqlBackend,
        schema: Schema,
        linker: EntityLinker,
        labels: LabelStore,
        top_k: int = 8,
        responses_per_entity: int = 10,
    ):
        self.top_k = top_k
        self.docs: list[tuple[str, str, str]] = []  # (status, text_for_index, text_for_evidence)
        ph = schema.uri
        structural = backend.select(
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(isIRI(?o)) "
            f"FILTER(?p NOT IN (<{RDF_TYPE}>, <{RDFS_LABEL}>, <{ph('ofDrug')}>, <{ph('ofCellLine')}>)) }}"
        )
        labels.prefetch([x for r in structural for x in (r["s"], r["o"])])
        facts: dict[str, list[str]] = {}
        for r in structural:
            facts.setdefault(r["s"], []).append(f"{schema.local(r['p'])} {labels.label(r['o'])}")
        for uri, label, cls in linker.entities:
            lines = [f"{cls} {label}."] + [f"{label} {f}." for f in sorted(facts.get(uri, []))]
            if cls in ("Drug", "CellLine"):
                lines += self._response_lines(backend, schema, labels, uri, cls, responses_per_entity)
            text = " ".join(lines)
            self.docs.append(("structural", text, text))
        self._vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform([d[1] for d in self.docs]) if self.docs else None

    @staticmethod
    def _response_lines(backend, schema, labels, uri, cls, k) -> list[str]:
        ph = schema.uri
        anchor = "ofDrug" if cls == "Drug" else "ofCellLine"
        other = "ofCellLine" if cls == "Drug" else "ofDrug"
        rows = backend.select(
            f"SELECT ?o ?auc WHERE {{ ?r a <{ph('Response')}> ; <{ph(anchor)}> <{uri}> ; "
            f"<{ph(other)}> ?o ; <{ph('auc')}> ?auc }} ORDER BY ?auc LIMIT {k}"
        )
        labels.prefetch([r["o"] for r in rows])
        return [f"MEASURED: {labels.label(uri)} with {labels.label(r['o'])}, AUC={r['auc']}." for r in rows]

    def retrieve(self, question: str) -> Retrieval:
        with Timer() as t:
            if self._matrix is None:
                return Retrieval(self.name, error="empty index")
            sims = (self._vectorizer.transform([question]) @ self._matrix.T).toarray()[0]
            top = sims.argsort()[::-1][: self.top_k]
            items = [(self.docs[i][0], self.docs[i][2]) for i in top if sims[i] > 0]
        return Retrieval(self.name, evidence=number(items), latency_s=t.elapsed)
