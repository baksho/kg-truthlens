"""Entity linking: map mentions in a question to KG entities.

Exact (normalised) label match first, then character-n-gram TF-IDF similarity for typos and
spelling variants. Linking is a known weak point of GraphRAG on biomedical KGs, so the eval
harness reports linking accuracy separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from graphrag_pharmaco.kg.backend import SparqlBackend
from graphrag_pharmaco.kg.schema import ENTITY_CLASSES, RDF_TYPE, RDFS_LABEL, Schema

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-+/.][A-Za-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or the to which what who whom "
    "that this these those with was were most least top lowest highest cell line lines drug "
    "drugs gene genes target targets targeted pathway pathways tissue sensitive sensitivity "
    "auc below above than not do does how many".split()
)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass(frozen=True)
class LinkedEntity:
    uri: str
    label: str
    cls: str
    mention: str
    score: float


class EntityLinker:
    def __init__(self, backend: SparqlBackend, schema: Schema, threshold: float = 0.85, max_ngram: int = 4):
        self.schema = schema
        self.threshold = threshold
        self.max_ngram = max_ngram
        values = " ".join(f"<{schema.uri(c)}>" for c in ENTITY_CLASSES)
        rows = backend.select(
            f"SELECT ?e ?l ?c WHERE {{ VALUES ?c {{ {values} }} ?e <{RDF_TYPE}> ?c ; <{RDFS_LABEL}> ?l }}"
        )
        self.entities: list[tuple[str, str, str]] = sorted(
            (r["e"], r["l"], schema.local(r["c"])) for r in rows
        )
        self._norm = [normalize(label) for _, label, _ in self.entities]
        self._exact: dict[str, int] = {}
        for i, n in enumerate(self._norm):
            self._exact.setdefault(n, i)
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if self.entities:
            self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
            self._matrix = self._vectorizer.fit_transform(self._norm)

    def _spans(self, text: str) -> list[tuple[int, int, str]]:
        tokens = [m.group(0) for m in _TOKEN.finditer(text)]
        spans = []
        for i in range(len(tokens)):
            for j in range(i + 1, min(len(tokens), i + self.max_ngram) + 1):
                span = " ".join(tokens[i:j])
                norm = normalize(span)
                words = norm.split()
                if not norm or all(w in STOPWORDS for w in words):
                    continue
                spans.append((i, j, span))
        return spans

    def link(self, text: str) -> list[LinkedEntity]:
        if not self.entities:
            return []
        spans = self._spans(text)
        candidates: list[tuple[float, int, int, int, int, str]] = []  # score, ntok, i, j, idx, span
        fuzzy: list[tuple[int, int, str]] = []
        for i, j, span in spans:
            idx = self._exact.get(normalize(span))
            if idx is not None:
                candidates.append((1.0, j - i, i, j, idx, span))
            elif len(normalize(span)) >= 4:
                fuzzy.append((i, j, span))
        if fuzzy and self._vectorizer is not None:
            sims = (self._vectorizer.transform([normalize(s) for _, _, s in fuzzy]) @ self._matrix.T).toarray()
            for row, (i, j, span) in zip(sims, fuzzy, strict=True):
                best = int(row.argmax())
                if row[best] >= self.threshold:
                    candidates.append((float(row[best]), j - i, i, j, best, span))
        candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
        taken: set[int] = set()
        chosen: list[LinkedEntity] = []
        for score, _, i, j, idx, span in candidates:
            if any(k in taken for k in range(i, j)):
                continue
            taken.update(range(i, j))
            uri, label, cls = self.entities[idx]
            chosen.append(LinkedEntity(uri, label, cls, span, score))
        return chosen
