"""Label lookup and human-readable rendering of KG nodes."""

from __future__ import annotations

from collections.abc import Iterable

from graphrag_pharmaco.kg.backend import SparqlBackend, batches, is_uri
from graphrag_pharmaco.kg.schema import RDFS_LABEL, Schema


class LabelStore:
    def __init__(self, backend: SparqlBackend, schema: Schema):
        self.backend = backend
        self.schema = schema
        self._cache: dict[str, str] = {}

    def prefetch(self, uris: Iterable[str]) -> None:
        missing = sorted({u for u in uris if is_uri(u) and u not in self._cache})
        for chunk in batches(missing, 100):
            values = " ".join(f"<{u}>" for u in chunk)
            rows = self.backend.select(
                f"SELECT ?x ?l WHERE {{ VALUES ?x {{ {values} }} ?x <{RDFS_LABEL}> ?l }}"
            )
            for row in rows:
                self._cache.setdefault(row["x"], row["l"])

    def label(self, uri: str) -> str:
        if not is_uri(uri):
            return uri
        if uri not in self._cache:
            self.prefetch([uri])
        return self._cache.get(uri) or self.schema.local(uri)

    def render(self, value: str) -> str:
        """Label for URIs, the raw string for literals."""
        return self.label(value) if is_uri(value) else value
