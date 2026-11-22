"""Wiring: build backend, linker and retrievers from Settings; `ask` runs retrieval + generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from graphrag_pharmaco.config import Settings
from graphrag_pharmaco.generation import Answer, generate
from graphrag_pharmaco.kg.backend import CachedBackend, HttpBackend, RdflibBackend, SparqlBackend
from graphrag_pharmaco.kg.labels import LabelStore
from graphrag_pharmaco.kg.linking import EntityLinker
from graphrag_pharmaco.kg.schema import Schema
from graphrag_pharmaco.llm import LLM
from graphrag_pharmaco.retrieval.base import Retrieval, Retriever
from graphrag_pharmaco.retrieval.hybrid import HybridRetriever
from graphrag_pharmaco.retrieval.neighborhood import NeighborhoodRetriever
from graphrag_pharmaco.retrieval.text2sparql import Text2SparqlRetriever
from graphrag_pharmaco.retrieval.vector import VectorRetriever

STRATEGIES = ("vector", "text2sparql", "neighborhood", "hybrid")


@dataclass
class System:
    backend: SparqlBackend
    schema: Schema
    labels: LabelStore
    linker: EntityLinker
    llm: LLM | None
    max_rows: int = 200
    _retrievers: dict[str, Retriever] = field(default_factory=dict)

    def retriever(self, strategy: str) -> Retriever:
        if strategy in self._retrievers:
            return self._retrievers[strategy]
        if strategy == "neighborhood":
            r: Retriever = NeighborhoodRetriever(self.backend, self.schema, self.linker, self.labels)
        elif strategy == "vector":
            r = VectorRetriever(self.backend, self.schema, self.linker, self.labels)
        elif strategy in ("text2sparql", "hybrid"):
            if self.llm is None:
                raise RuntimeError(f"strategy '{strategy}' needs an LLM")
            t2s = Text2SparqlRetriever(self.backend, self.schema, self.labels, self.llm, self.max_rows)
            if strategy == "text2sparql":
                r = t2s
            else:
                nb = NeighborhoodRetriever(self.backend, self.schema, self.linker, self.labels)
                r = HybridRetriever(self.linker, t2s, nb)
        else:
            raise ValueError(f"unknown strategy '{strategy}'; choose from {STRATEGIES}")
        self._retrievers[strategy] = r
        return r

    def ask(self, question: str, strategy: str = "hybrid") -> tuple[Retrieval, Answer]:
        retrieval = self.retriever(strategy).retrieve(question)
        if self.llm is None:
            return retrieval, Answer(abstain=True, note="no LLM configured")
        return retrieval, generate(question, retrieval, self.llm)


def build_system(settings: Settings, llm: LLM | None = None, backend: SparqlBackend | None = None) -> System:
    schema = Schema(settings.kg_ns)
    if backend is None:
        if settings.kg_backend == "http":
            if not settings.sparql_endpoint:
                raise RuntimeError("Set GRAPHRAG_SPARQL_ENDPOINT for the http backend.")
            inner: SparqlBackend = HttpBackend(settings.sparql_endpoint, settings.sparql_timeout_s)
        elif settings.kg_backend == "rdflib":
            if not settings.kg_files:
                raise RuntimeError("Set GRAPHRAG_KG_FILES (comma-separated RDF files) for the rdflib backend.")
            inner = RdflibBackend.from_files(settings.kg_files)
        else:
            raise RuntimeError(f"Unknown GRAPHRAG_KG_BACKEND: {settings.kg_backend}")
        backend = CachedBackend(inner, settings.cache_size, settings.cache_ttl_s)
    labels = LabelStore(backend, schema)
    return System(backend, schema, labels, EntityLinker(backend, schema), llm, settings.max_rows)
