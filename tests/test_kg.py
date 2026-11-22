import pytest

from graphrag_pharmaco.cache import TTLCache
from graphrag_pharmaco.kg.guard import QueryRejected, extract_sparql, guard


def test_guard_accepts_select_and_adds_limit():
    q = guard("SELECT ?s WHERE { ?s ?p ?o }", max_rows=50)
    assert q.endswith("LIMIT 50")


def test_guard_clamps_large_limit_and_keeps_small():
    assert guard("SELECT ?s WHERE { ?s ?p ?o } LIMIT 100000", 200).endswith("LIMIT 200")
    assert guard("SELECT ?s WHERE { ?s ?p ?o } LIMIT 5", 200).endswith("LIMIT 5")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "SELECT ?s WHERE {",
        "DELETE WHERE { ?s ?p ?o }",
        "INSERT DATA { <a:b> <a:c> <a:d> }",
        "ASK { ?s ?p ?o }",
        "SELECT * WHERE { SERVICE <http://evil.example/sparql> { ?s ?p ?o } }",
    ],
)
def test_guard_rejects(bad):
    with pytest.raises(QueryRejected):
        guard(bad)


def test_extract_sparql_from_fence():
    assert extract_sparql("blah\n```sparql\nSELECT ?x WHERE { ?x ?p ?o }\n```\nbye") == "SELECT ?x WHERE { ?x ?p ?o }"
    assert extract_sparql("SELECT ?x WHERE { ?x ?p ?o }") == "SELECT ?x WHERE { ?x ?p ?o }"


def test_ttl_cache_expiry_and_lru():
    now = [0.0]
    c: TTLCache[int] = TTLCache(maxsize=2, ttl_s=10, clock=lambda: now[0])
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)  # evicts b (least recently used)
    assert c.get("b") is None
    now[0] = 11
    assert c.get("a") is None


def test_linker_exact_fuzzy_and_stopwords(system):
    linked = system.linker.link("Which genes does testdrugalpha target?")
    assert [e.label for e in linked] == ["TestDrugAlpha"]
    fuzzy = system.linker.link("what does TestDrugAlpa do")  # one dropped letter
    assert [e.label for e in fuzzy] == ["TestDrugAlpha"]
    # a different, similarly named entity must not be linked at the default threshold
    assert system.linker.link("what does TestDrugGamma do") == []
    assert system.linker.link("Which genes and which drugs are there?") == []
    both = {e.label for e in system.linker.link("Lung cell lines for TestDrugBeta")}
    assert both == {"Lung", "TestDrugBeta"}


def test_labels_fall_back_to_local_name(system):
    assert system.labels.label("https://example.org/pharmaco#alpha") == "TestDrugAlpha"
    assert system.labels.label("https://example.org/pharmaco#nolabel") == "nolabel"
    assert system.labels.render("0.5") == "0.5"


def test_with_prefixes_adds_only_missing(schema):
    q = schema.with_prefixes("SELECT ?x WHERE { ?x a ph:Drug }")
    assert q.count("PREFIX ph:") == 1 and "PREFIX rdfs:" in q
    declared = "PREFIX ph: <urn:custom#>\nSELECT ?x WHERE { ?x a ph:Drug }"
    assert schema.with_prefixes(declared).count("PREFIX ph:") == 1
