from graphrag_pharmaco.generation import generate, parse_answer
from graphrag_pharmaco.kg.guard import QueryRejected  # noqa: F401  (import check)
from graphrag_pharmaco.llm import ScriptedLLM
from graphrag_pharmaco.retrieval.hybrid import HybridRetriever
from graphrag_pharmaco.retrieval.neighborhood import NeighborhoodRetriever
from graphrag_pharmaco.retrieval.text2sparql import Text2SparqlRetriever
from graphrag_pharmaco.retrieval.vector import VectorRetriever


def texts(retrieval):
    return "\n".join(e.text for e in retrieval.evidence)


def test_neighborhood_collapses_responses_and_labels_status(system):
    r = NeighborhoodRetriever(system.backend, system.schema, system.linker, system.labels).retrieve(
        "Tell me about TestDrugAlpha"
    )
    t = texts(r)
    assert "TestDrugAlpha --targets--> TGT1" in t
    assert "MEASURED: TestDrugAlpha on CL-1, AUC=0.21" in t
    assert "PREDICTED (model output, not a measurement): TestDrugAlpha on CL-1" in t
    assert "resp_a1" not in t  # raw response-node triples are collapsed
    assert {e.status for e in r.evidence} == {"structural", "measured", "predicted"}


def test_vector_baseline_retrieves_relevant_doc(system):
    r = VectorRetriever(system.backend, system.schema, system.linker, system.labels).retrieve(
        "Which genes does TestDrugBeta target"
    )
    assert "TestDrugBeta" in r.evidence[0].text


def _t2s(system, replies):
    it = iter(replies)
    llm = ScriptedLLM(lambda s, u: next(it))
    return Text2SparqlRetriever(system.backend, system.schema, system.labels, llm), llm


GOOD = """```sparql
SELECT ?answer WHERE { ?d rdfs:label ?dl . FILTER(LCASE(STR(?dl)) = "testdrugalpha")
  ?d ph:targets ?g . ?g rdfs:label ?answer }
```"""


def test_text2sparql_runs_valid_query(system):
    r, _ = _t2s(system, [GOOD])
    out = r.retrieve("Which genes does TestDrugAlpha target?")
    assert out.sparql_valid is True and out.error is None
    assert sorted(e.text for e in out.evidence) == ["answer: TGT1", "answer: TGT2"]


def test_text2sparql_retries_after_rejected_query_and_reports_feedback(system):
    r, llm = _t2s(system, ["DELETE WHERE { ?s ?p ?o }", GOOD])
    out = r.retrieve("Which genes does TestDrugAlpha target?")
    assert out.sparql_valid is True
    assert "rejected" in llm.calls[1][1]


def test_text2sparql_gives_up_and_reports_invalid(system):
    r, _ = _t2s(system, ["nonsense"] * 3)
    out = r.retrieve("anything")
    assert out.sparql_valid is False and out.evidence == [] and out.error


def test_text2sparql_marks_predicted_status(system):
    q = """```sparql
SELECT ?answer WHERE { ?r a ph:PredictedResponse ; ph:ofCellLine ?c ; ph:score ?s . ?c rdfs:label ?answer }
ORDER BY DESC(?s) LIMIT 2
```"""
    r, _ = _t2s(system, [q])
    out = r.retrieve("x")
    assert [e.status for e in out.evidence] == ["predicted", "predicted"]


def test_hybrid_passes_uris_and_falls_back_to_neighborhood(system):
    seen = {}

    def fn(s, u):
        seen["user"] = u
        return "```sparql\nSELECT ?answer WHERE { ?x ph:doesNotExist ?answer }\n```"  # valid, empty

    t2s = Text2SparqlRetriever(system.backend, system.schema, system.labels, ScriptedLLM(fn), max_attempts=1)
    nb = NeighborhoodRetriever(system.backend, system.schema, system.linker, system.labels)
    out = HybridRetriever(system.linker, t2s, nb).retrieve("Which genes does TestDrugAlpha target?")
    assert "https://example.org/pharmaco#alpha" in seen["user"]
    assert out.strategy == "hybrid" and "TestDrugAlpha --targets--> TGT1" in texts(out)


def test_parse_answer_variants():
    ok = parse_answer('Sure! {"answer": ["A"], "citations": ["E1"], "abstain": false, "note": ""}')
    assert ok.parse_ok and ok.answer == ["A"]
    assert not parse_answer("no json here").parse_ok
    assert not parse_answer("{broken").parse_ok


def test_generate_abstains_without_evidence_and_skips_llm(system):
    from graphrag_pharmaco.retrieval.base import Retrieval

    llm = ScriptedLLM(lambda s, u: "should not be called")
    a = generate("q", Retrieval("x"), llm)
    assert a.abstain and llm.calls == []
