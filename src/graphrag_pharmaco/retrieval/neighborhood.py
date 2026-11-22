"""Strategy: link entities in the question, then expand their k-hop neighbourhood."""

from __future__ import annotations

from graphrag_pharmaco.kg.backend import SparqlBackend, is_uri
from graphrag_pharmaco.kg.guard import guard
from graphrag_pharmaco.kg.labels import LabelStore
from graphrag_pharmaco.kg.linking import EntityLinker, LinkedEntity
from graphrag_pharmaco.kg.schema import RDF_TYPE, RDFS_LABEL, Schema
from graphrag_pharmaco.kg.verbalize import Verbalizer
from graphrag_pharmaco.retrieval.base import Retrieval, Timer, number


class NeighborhoodRetriever:
    name = "neighborhood"

    def __init__(
        self,
        backend: SparqlBackend,
        schema: Schema,
        linker: EntityLinker,
        labels: LabelStore,
        hops: int = 2,
        max_triples: int = 150,
    ):
        self.backend = backend
        self.schema = schema
        self.linker = linker
        self.labels = labels
        self.verbalizer = Verbalizer(backend, schema, labels)
        self.hops = hops
        self.max_triples = max_triples

    def retrieve(self, question: str) -> Retrieval:
        with Timer() as t:
            linked = self.linker.link(question)
            result = self.expand(linked)
        result.latency_s = t.elapsed
        return result

    def expand(self, linked: list[LinkedEntity], strategy: str | None = None) -> Retrieval:
        triples: dict[tuple[str, str, str], None] = {}
        visited: set[str] = set()
        frontier = [e.uri for e in linked]
        error = None
        try:
            for _ in range(self.hops):
                frontier = [u for u in frontier if u not in visited]
                if not frontier or len(triples) >= self.max_triples:
                    break
                visited.update(frontier)
                values = " ".join(f"<{u}>" for u in frontier)
                query = (
                    f"SELECT ?s ?p ?o WHERE {{ {{ VALUES ?s {{ {values} }} ?s ?p ?o }} "
                    f"UNION {{ VALUES ?o {{ {values} }} ?s ?p ?o }} "
                    f"FILTER(?p NOT IN (<{RDF_TYPE}>, <{RDFS_LABEL}>)) }}"
                )
                new_nodes: set[str] = set()
                for row in self.backend.select(guard(query, self.max_triples)):
                    triples[(row["s"], row["p"], row["o"])] = None
                    new_nodes.update(x for x in (row["s"], row["o"]) if is_uri(x))
                frontier = sorted(new_nodes)
        except Exception as exc:  # backend failure is reported, not raised
            error = f"{type(exc).__name__}: {exc}"

        response_preds = {self.schema.uri(p) for p in ("ofDrug", "ofCellLine", "auc", "score")}
        response_nodes = sorted({s for (s, p, _o) in triples if p in response_preds})
        collapsed = self.verbalizer.response_nodes(response_nodes) if response_nodes else {}
        self.labels.prefetch([x for tr in triples for x in tr])
        items: list[tuple[str, str]] = []
        for s, p, o in triples:
            if s in collapsed and p in response_preds:
                continue  # represented by the collapsed MEASURED/PREDICTED line below
            items.append(("structural", self.verbalizer.triple_text(s, p, o)))
        items.extend(collapsed.values())
        return Retrieval(
            strategy or self.name,
            evidence=number(items),
            linked=[e.uri for e in linked],
            error=error,
        )
