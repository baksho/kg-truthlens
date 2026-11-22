"""Strategy: entity linking -> schema-aware text-to-SPARQL with resolved URIs -> neighbourhood fallback."""

from __future__ import annotations

from graphrag_pharmaco.kg.linking import EntityLinker
from graphrag_pharmaco.retrieval.base import Retrieval, Timer
from graphrag_pharmaco.retrieval.neighborhood import NeighborhoodRetriever
from graphrag_pharmaco.retrieval.text2sparql import Text2SparqlRetriever


class HybridRetriever:
    name = "hybrid"

    def __init__(self, linker: EntityLinker, text2sparql: Text2SparqlRetriever, neighborhood: NeighborhoodRetriever):
        self.linker = linker
        self.text2sparql = text2sparql
        self.neighborhood = neighborhood

    def retrieve(self, question: str) -> Retrieval:
        with Timer() as t:
            linked = self.linker.link(question)
            result = self.text2sparql.retrieve_with_hints(question, linked, strategy=self.name)
            if not result.evidence and linked:
                fallback = self.neighborhood.expand(linked, strategy=self.name)
                fallback.sparql, fallback.sparql_valid = result.sparql, result.sparql_valid
                fallback.error = result.error or fallback.error
                result = fallback
        result.latency_s = t.elapsed
        return result
