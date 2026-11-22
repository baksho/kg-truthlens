"""SPARQL backends: in-memory rdflib (tests, small KGs) and HTTP (GraphDB / any SPARQL endpoint)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import httpx
from rdflib import Graph

from graphrag_pharmaco.cache import TTLCache

Row = dict[str, str]

_SAFE_IRI = re.compile(r'^[^\s<>"{}|\\^`]+$')


def is_uri(value: str) -> bool:
    return value.startswith(("http://", "https://", "urn:")) and bool(_SAFE_IRI.match(value))


class SparqlBackend(Protocol):
    def select(self, query: str) -> list[Row]:
        """Run a SELECT query. Values are returned as plain strings (URIs unwrapped)."""
        ...


class RdflibBackend:
    """In-memory graph. Note: rdflib cannot enforce query timeouts; use HttpBackend for that."""

    def __init__(self, graph: Graph | None = None):
        self.graph = graph if graph is not None else Graph()

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> RdflibBackend:
        graph = Graph()
        for path in paths:
            graph.parse(str(path))  # format guessed from the file extension
        return cls(graph)

    def select(self, query: str) -> list[Row]:
        result = self.graph.query(query)
        return [{str(k): str(v) for k, v in row.asdict().items()} for row in result]


class HttpBackend:
    """SPARQL 1.1 protocol over HTTP (POST, form-encoded)."""

    def __init__(self, endpoint: str, timeout_s: float = 30.0, client: httpx.Client | None = None):
        self.endpoint = endpoint
        self.client = client or httpx.Client(timeout=timeout_s)

    def select(self, query: str) -> list[Row]:
        response = self.client.post(
            self.endpoint,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
        )
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
        return [{k: v["value"] for k, v in binding.items()} for binding in bindings]


class CachedBackend:
    """Wraps a backend with a TTL/LRU cache keyed on the exact query text."""

    def __init__(self, inner: SparqlBackend, maxsize: int = 512, ttl_s: float = 600.0):
        self.inner = inner
        self.cache: TTLCache[list[Row]] = TTLCache(maxsize, ttl_s)

    def select(self, query: str) -> list[Row]:
        hit = self.cache.get(query)
        if hit is not None:
            return hit
        rows = self.inner.select(query)
        self.cache.put(query, rows)
        return rows


def batches(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
