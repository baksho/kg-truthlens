"""Benchmark generation: question templates whose gold answers are computed from the KG itself.

No manual annotation and no invented data: every gold answer is the result of a SPARQL query on
the real graph. Unanswerable questions ask about things the schema does not contain (doses,
approval status), so the correct behaviour is to abstain.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from graphrag_pharmaco.kg.backend import SparqlBackend
from graphrag_pharmaco.kg.schema import Schema

MAX_GOLD = 25


@dataclass
class Question:
    id: str
    kind: str  # single_hop | reverse_hop | two_hop | ranking | filtered | predicted | unanswerable
    question: str
    gold: list[str]
    answerable: bool
    gold_sparql: str = ""
    entities: tuple[str, ...] = ()  # URIs a perfect linker should find

    def to_dict(self) -> dict:
        return asdict(self)


def _sample(rows: list, n: int, rng: random.Random) -> list:
    rows = sorted(rows, key=lambda r: tuple(sorted(r.items())))
    return rng.sample(rows, min(n, len(rows)))


def generate_questions(
    backend: SparqlBackend, schema: Schema, n_per_template: int = 10, seed: int = 0
) -> list[Question]:
    rng = random.Random(seed)
    P = schema.prefix
    out: list[Question] = []

    def gold_of(query: str) -> list[str]:
        return sorted({r["answer"] for r in backend.select(P + query)})

    def add(kind: str, question: str, gold_sparql: str, entities: tuple[str, ...], ordered: bool = False):
        gold = [r["answer"] for r in backend.select(P + gold_sparql)] if ordered else gold_of(gold_sparql)
        if not gold or len(gold) > MAX_GOLD:
            return
        out.append(Question(f"{kind}-{sum(q.kind == kind for q in out)}", kind, question, gold, True, gold_sparql, entities))

    # 1. single hop: drug -> genes
    for r in _sample(backend.select(P + "SELECT DISTINCT ?d ?dl WHERE { ?d ph:targets ?g ; rdfs:label ?dl }"), n_per_template, rng):
        add("single_hop", f"Which genes does {r['dl']} target?",
            f"SELECT DISTINCT ?answer WHERE {{ <{r['d']}> ph:targets ?g . ?g rdfs:label ?answer }}", (r["d"],))

    # 2. reverse hop: gene -> drugs
    for r in _sample(backend.select(P + "SELECT DISTINCT ?g ?gl WHERE { ?d ph:targets ?g . ?g rdfs:label ?gl }"), n_per_template, rng):
        add("reverse_hop", f"Which drugs target {r['gl']}?",
            f"SELECT DISTINCT ?answer WHERE {{ ?d ph:targets <{r['g']}> ; rdfs:label ?answer }}", (r["g"],))

    # 3. two hop: pathway -> drugs -> genes
    for r in _sample(backend.select(P + "SELECT DISTINCT ?p ?pl WHERE { ?d ph:targetPathway ?p . ?p rdfs:label ?pl }"), n_per_template, rng):
        add("two_hop", f"Which genes are targeted by drugs in the {r['pl']} pathway?",
            f"SELECT DISTINCT ?answer WHERE {{ ?d ph:targetPathway <{r['p']}> ; ph:targets ?g . ?g rdfs:label ?answer }}",
            (r["p"],))

    drugs_with_resp = backend.select(
        P + "SELECT DISTINCT ?d ?dl WHERE { ?r a ph:Response ; ph:ofDrug ?d . ?d rdfs:label ?dl }"
    )

    # 4. ranking: 5 most sensitive cell lines (lowest AUC); skip if there is a tie at the cut
    for r in _sample(drugs_with_resp, n_per_template, rng):
        aucs = backend.select(P + f"SELECT ?auc WHERE {{ ?r a ph:Response ; ph:ofDrug <{r['d']}> ; ph:auc ?auc }} ORDER BY ?auc LIMIT 6")
        if len(aucs) < 6 or aucs[4]["auc"] == aucs[5]["auc"]:
            continue
        add("ranking", f"Which 5 cell lines are most sensitive to {r['dl']} (lowest AUC)?",
            f"SELECT ?answer WHERE {{ ?r a ph:Response ; ph:ofDrug <{r['d']}> ; ph:ofCellLine ?c ; ph:auc ?auc . "
            f"?c rdfs:label ?answer }} ORDER BY ?auc LIMIT 5", (r["d"],), ordered=True)

    # 5. filtered multi-hop: cell lines of a tissue below the drug's median AUC
    for r in _sample(drugs_with_resp, n_per_template, rng):
        vals = sorted(float(x["auc"]) for x in backend.select(
            P + f"SELECT ?auc WHERE {{ ?r a ph:Response ; ph:ofDrug <{r['d']}> ; ph:auc ?auc }}"))
        if len(vals) < 4:
            continue
        thr = round(vals[len(vals) // 2], 3)
        tissues = backend.select(P + f"SELECT DISTINCT ?t ?tl WHERE {{ ?r a ph:Response ; ph:ofDrug <{r['d']}> ; ph:ofCellLine ?c . ?c ph:tissue ?t . ?t rdfs:label ?tl }}")
        if not tissues:
            continue
        t = rng.choice(sorted(tissues, key=lambda x: x["t"]))
        add("filtered", f"Which {t['tl']} cell lines have an AUC below {thr} for {r['dl']}?",
            f"SELECT DISTINCT ?answer WHERE {{ ?r a ph:Response ; ph:ofDrug <{r['d']}> ; ph:ofCellLine ?c ; ph:auc ?auc . "
            f"FILTER(?auc < {thr}) ?c ph:tissue <{t['t']}> ; rdfs:label ?answer }}", (r["d"], t["t"]))

    # 6. predicted vs measured: top-5 predicted scores (probes observed/predicted separation)
    pred_drugs = backend.select(P + "SELECT DISTINCT ?d ?dl WHERE { ?r a ph:PredictedResponse ; ph:ofDrug ?d . ?d rdfs:label ?dl }")
    for r in _sample(pred_drugs, n_per_template, rng):
        scores = backend.select(P + f"SELECT ?s WHERE {{ ?r a ph:PredictedResponse ; ph:ofDrug <{r['d']}> ; ph:score ?s }} ORDER BY DESC(?s) LIMIT 6")
        if len(scores) < 6 or scores[4]["s"] == scores[5]["s"]:
            continue
        add("predicted", f"According to the model predictions (not measurements), which 5 cell lines have the highest predicted response score for {r['dl']}?",
            f"SELECT ?answer WHERE {{ ?r a ph:PredictedResponse ; ph:ofDrug <{r['d']}> ; ph:ofCellLine ?c ; ph:score ?s . "
            f"?c rdfs:label ?answer }} ORDER BY DESC(?s) LIMIT 5", (r["d"],), ordered=True)

    # 7. unanswerable: asks for facts the KG schema does not hold
    all_drugs = backend.select(P + "SELECT DISTINCT ?d ?dl WHERE { ?d a ph:Drug ; rdfs:label ?dl }")
    patterns = [
        "What is the maximum tolerated dose of {x}?",
        "What is the clinical approval status of {x}?",
        "Which clinical trials has {x} been tested in?",
    ]
    for i, r in enumerate(_sample(all_drugs, n_per_template, rng)):
        out.append(Question(f"unanswerable-{i}", "unanswerable", patterns[i % len(patterns)].format(x=r["dl"]),
                            [], False, "", (r["d"],)))
    return out
