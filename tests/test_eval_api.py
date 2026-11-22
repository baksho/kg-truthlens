import json

from fastapi.testclient import TestClient

from graphrag_pharmaco.api import create_app
from graphrag_pharmaco.eval.metrics import score_answer
from graphrag_pharmaco.eval.questions import generate_questions
from graphrag_pharmaco.eval.run import run_eval, summarize
from graphrag_pharmaco.generation import Answer
from graphrag_pharmaco.llm import ScriptedLLM
from graphrag_pharmaco.retrieval.base import Evidence, Retrieval
from tests.conftest import ALPHA_AUC  # noqa: F401


def questions(system):
    return generate_questions(system.backend, system.schema, n_per_template=10, seed=0)


def test_generated_gold_matches_the_graph(system):
    qs = {q.kind: q for q in questions(system)}
    assert {"single_hop", "reverse_hop", "two_hop", "ranking", "filtered", "predicted", "unanswerable"} <= set(qs)
    alpha_rank = next(q for q in questions(system) if q.kind == "ranking" and "TestDrugAlpha" in q.question)
    assert alpha_rank.gold == ["CL-1", "CL-2", "CL-3", "CL-4", "CL-5"]  # lowest AUC first
    assert qs["unanswerable"].gold == [] and not qs["unanswerable"].answerable
    filt = next(q for q in questions(system) if q.kind == "filtered" and "TestDrugAlpha" in q.question)
    assert sorted(filt.gold) == ["CL-1", "CL-2", "CL-3"] and "Lung" in filt.question


def test_generation_is_deterministic_per_seed(system):
    a = [q.question for q in questions(system)]
    b = [q.question for q in questions(system)]
    assert a == b


def test_metrics_catch_unsupported_answers_and_bad_citations(system):
    q = next(q for q in questions(system) if q.kind == "single_hop" and "TestDrugBeta" in q.question)
    r = Retrieval("x", evidence=[Evidence("E1", "TestDrugBeta --targets--> TGT2")])
    good = score_answer(q, r, Answer(["TGT2"], ["E1"]))
    assert good.f1 == 1.0 and good.unsupported_rate == 0.0 and good.citation_validity == 1.0
    bad = score_answer(q, r, Answer(["TGT2", "EGFR"], ["E1", "E9"]))
    assert bad.precision == 0.5 and bad.unsupported_rate == 0.5 and bad.citation_validity == 0.5


def test_unanswerable_scoring(system):
    q = next(q for q in questions(system) if q.kind == "unanswerable")
    r = Retrieval("x", evidence=[Evidence("E1", "something")])
    assert score_answer(q, r, Answer(abstain=True)).abstain_correct == 1.0
    assert score_answer(q, r, Answer(["made up"], ["E1"])).abstain_correct == 0.0


def test_retrieval_only_eval_separates_strategies(system):
    qs = questions(system)
    records = run_eval(system, qs, ["neighborhood", "vector"], retrieval_only=True)
    table = summarize(records)
    nb = table["neighborhood"]["ALL"]["retrieval_recall"]
    assert nb is not None and nb > 0.5
    assert table["neighborhood"]["single_hop"]["linking_recall"] == 1.0


def test_end_to_end_with_scripted_llm(system):
    def llm_fn(sys_prompt, user):
        if "Evidence:" in user:
            return json.dumps({"answer": ["TGT1", "TGT2"], "citations": ["E1", "E2"], "abstain": False, "note": ""})
        return "```sparql\nSELECT ?answer WHERE { ?d ph:targets ?g . ?g rdfs:label ?answer ; ^ph:targets ph:alpha }\n```"

    system.llm = ScriptedLLM(llm_fn)
    system._retrievers.clear()
    try:
        retrieval, answer = system.ask("Which genes does TestDrugAlpha target?", "neighborhood")
        assert answer.parse_ok and answer.answer == ["TGT1", "TGT2"]
    finally:
        system.llm = None
        system._retrievers.clear()


def test_api_ask_health_validation_and_cache(system):
    system.llm = ScriptedLLM(
        lambda s, u: json.dumps({"answer": ["TGT2"], "citations": ["E1"], "abstain": False, "note": ""})
    )
    system._retrievers.clear()
    try:
        client = TestClient(create_app(system=system))
        assert client.get("/health").json()["status"] == "ok"
        body = {"question": "Which genes does TestDrugBeta target?", "strategy": "neighborhood"}
        first = client.post("/ask", json=body).json()
        assert first["answer"] == ["TGT2"] and first["cached"] is False and first["evidence"]
        assert client.post("/ask", json=body).json()["cached"] is True
        assert client.post("/ask", json={"question": "x y z", "strategy": "nope"}).status_code == 422
        assert client.post("/ask", json={"question": "x"}).status_code == 422
    finally:
        system.llm = None
        system._retrievers.clear()


def test_api_without_llm_returns_503_for_llm_strategies(system):
    client = TestClient(create_app(system=system))
    r = client.post("/ask", json={"question": "Which genes does TestDrugBeta target?", "strategy": "hybrid"})
    assert r.status_code == 503


def test_cli_retrieval_only_from_rdf_file(tmp_path, capsys):
    from graphrag_pharmaco.eval.run import main
    from tests.conftest import build_graph

    kg = tmp_path / "kg.ttl"
    build_graph().serialize(str(kg), format="turtle")
    out = tmp_path / "run"
    main(["--kg", str(kg), "--strategies", "neighborhood,vector", "--retrieval-only", "--out", str(out)])
    assert (out / "summary.md").exists() and (out / "questions.json").exists()
    assert "neighborhood" in capsys.readouterr().out
