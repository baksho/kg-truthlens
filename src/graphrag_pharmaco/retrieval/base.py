from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Evidence:
    id: str
    text: str
    status: str = "measured"  # "measured" | "predicted" | "structural"


@dataclass
class Retrieval:
    strategy: str
    evidence: list[Evidence] = field(default_factory=list)
    sparql: str | None = None
    linked: list[str] = field(default_factory=list)  # URIs of linked entities
    error: str | None = None
    sparql_valid: bool | None = None  # None = strategy does not write SPARQL
    latency_s: float = 0.0


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str) -> Retrieval: ...


def number(items: list[tuple[str, str]]) -> list[Evidence]:
    """[(status, text), ...] -> Evidence with ids E1, E2, ... (deduplicated, order kept)."""
    seen: set[str] = set()
    out: list[Evidence] = []
    for status, text in items:
        if text in seen:
            continue
        seen.add(text)
        out.append(Evidence(f"E{len(out) + 1}", text, status))
    return out


class Timer:
    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self.start
