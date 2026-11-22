"""TEST FIXTURE ONLY.

A hand-built miniature graph that follows docs/KG_CONTRACT.md so the code paths can be unit
tested offline. Names and numbers are placeholders, not real measurements, and must never be
used for evaluation or reported results. Real evaluation runs against the KG repo's export.
"""

from __future__ import annotations

import pytest
from rdflib import RDF, RDFS, Graph, Literal, Namespace

from graphrag_pharmaco.kg.backend import RdflibBackend
from graphrag_pharmaco.kg.labels import LabelStore
from graphrag_pharmaco.kg.linking import EntityLinker
from graphrag_pharmaco.kg.schema import Schema
from graphrag_pharmaco.system import System

PH = Namespace("https://example.org/pharmaco#")

ALPHA_AUC = [0.21, 0.35, 0.44, 0.52, 0.63, 0.71, 0.88]
BETA_AUC = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
ALPHA_PRED = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]


def build_graph() -> Graph:
    g = Graph()

    def node(uri, cls, label):
        g.add((uri, RDF.type, cls))
        g.add((uri, RDFS.label, Literal(label)))

    for name, cls in [("PathX", PH.Pathway), ("PathY", PH.Pathway), ("Lung", PH.Tissue), ("Breast", PH.Tissue),
                      ("TGT1", PH.Gene), ("TGT2", PH.Gene)]:
        node(PH[name], cls, name)
    node(PH.alpha, PH.Drug, "TestDrugAlpha")
    node(PH.beta, PH.Drug, "TestDrugBeta")
    g.add((PH.alpha, PH.targets, PH.TGT1))
    g.add((PH.alpha, PH.targets, PH.TGT2))
    g.add((PH.alpha, PH.targetPathway, PH.PathX))
    g.add((PH.beta, PH.targets, PH.TGT2))
    g.add((PH.beta, PH.targetPathway, PH.PathY))
    for i in range(1, 8):
        cl = PH[f"cl{i}"]
        node(cl, PH.CellLine, f"CL-{i}")
        g.add((cl, PH.tissue, PH.Lung if i <= 3 else PH.Breast))
        for drug, aucs, tag in ((PH.alpha, ALPHA_AUC, "a"), (PH.beta, BETA_AUC, "b")):
            r = PH[f"resp_{tag}{i}"]
            g.add((r, RDF.type, PH.Response))
            g.add((r, PH.ofDrug, drug))
            g.add((r, PH.ofCellLine, cl))
            g.add((r, PH.auc, Literal(aucs[i - 1])))
        p = PH[f"pred_a{i}"]
        g.add((p, RDF.type, PH.PredictedResponse))
        g.add((p, PH.ofDrug, PH.alpha))
        g.add((p, PH.ofCellLine, cl))
        g.add((p, PH.score, Literal(ALPHA_PRED[i - 1])))
    return g


@pytest.fixture(scope="session")
def backend() -> RdflibBackend:
    return RdflibBackend(build_graph())


@pytest.fixture(scope="session")
def schema() -> Schema:
    return Schema()


@pytest.fixture(scope="session")
def system(backend, schema) -> System:
    labels = LabelStore(backend, schema)
    return System(backend, schema, labels, EntityLinker(backend, schema), llm=None)
