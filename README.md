# graphrag-pharmaco

GraphRAG over a pharmacogenomics knowledge graph, with an evaluation harness that measures
whether answers stay grounded in the graph.

It compares four retrieval strategies on the same questions:

| Strategy | How evidence is found |
|---|---|
| `vector` | Baseline RAG: TF-IDF search over one text document per entity, built from the KG. |
| `text2sparql` | The LLM writes a SPARQL query from the schema; validated, executed, retried on errors. |
| `neighborhood` | Link entities in the question, then expand their k-hop neighbourhood. |
| `hybrid` | Entity linking → text-to-SPARQL with resolved URIs → neighbourhood fallback. |

Answers are structured JSON (`answer`, `citations`, `abstain`, `note`). **Measured** and
**predicted** drug responses are separate evidence types and are labelled in every evidence line.

## Status: what is and is not verified

- Verified (31 unit tests, offline): SPARQL guard, entity linking, all four retrievers, answer
  parsing, metrics, benchmark generation, eval CLI, API.
- **Not run on real data.** The authoring environment could not download GDSC2 and the unit
  tests use a tiny hand-built placeholder graph (`tests/conftest.py`). No result in this repo
  is a real result until you run the eval on your KG export.
- **Not run with a real LLM.** Tests use a scripted stand-in. `AnthropicLLM` and
  `OpenAICompatLLM` follow the documented request shapes but have not been called here.
- **Docker/GraphDB untested** (no Docker daemon available). `docker-compose.yml` and
  `scripts/load_graphdb.py` are written from the documented interfaces; verify them.
- Default entity-linking threshold (0.85) is an untuned guess. Tune it with the
  `linking_recall` metric on your real KG.

## Quick start

```bash
pip install -e ".[dev]"        # add ".[anthropic]" for the Anthropic adapter
pytest -q

# retrieval-only evaluation (no LLM needed for vector/neighborhood)
python -m graphrag_pharmaco.eval.run --kg data/kg.ttl --strategies vector,neighborhood \
    --retrieval-only --out runs/retrieval

# full evaluation: choose the model yourself (nothing is hard-coded)
export GRAPHRAG_LLM_PROVIDER=anthropic GRAPHRAG_LLM_MODEL=<model-id> ANTHROPIC_API_KEY=...
python -m graphrag_pharmaco.eval.run --kg data/kg.ttl --out runs/full

# serve
export GRAPHRAG_KG_FILES=data/kg.ttl      # or GRAPHRAG_KG_BACKEND=http + GRAPHRAG_SPARQL_ENDPOINT
uvicorn graphrag_pharmaco.api:app --port 8080
curl -s localhost:8080/ask -H 'content-type: application/json' \
  -d '{"question": "Which 5 cell lines are most sensitive to <drug>?", "strategy": "hybrid"}'
```

For a local model, use `GRAPHRAG_LLM_PROVIDER=openai_compat` with `GRAPHRAG_LLM_BASE_URL`
pointing at an OpenAI-style `/v1` server.

## Evaluation

`eval.questions` generates questions from templates and computes **gold answers by SPARQL on the
KG itself**, so no manual annotation and no invented data. Kinds: `single_hop`, `reverse_hop`,
`two_hop`, `ranking`, `filtered`, `predicted`, and `unanswerable` (asks for doses, approval
status or trials, which the graph does not hold; the right behaviour is to abstain).
Ranking and predicted questions are scored as sets of the top-k, not as ordered lists.

Metrics (`eval/metrics.py`): precision/recall/F1, `abstain_correct` (unanswerable),
`over_abstain` (answerable), `citation_validity`, `unsupported_rate` (answer items absent from
the evidence), `retrieval_recall` and `linking_recall` (no LLM needed), `sparql_valid`, latency.

Known limits: the unsupported-answer check is a string match on structured answers, not a
semantic entailment check; it will not catch a correct-looking answer that misreads the
evidence. Questions are template-based; paraphrase robustness is not measured yet.

## Layout

```
src/graphrag_pharmaco/
  kg/          backends (rdflib, HTTP), SPARQL guard, linking, labels, verbalizer, schema
  retrieval/   vector, text2sparql, neighborhood, hybrid
  generation.py  grounded JSON answers      eval/  benchmark, metrics, CLI
  api.py       FastAPI service (answer cache, 503 on misconfiguration)
docs/KG_CONTRACT.md   what this repo expects from the KG repo
```

## Safety notes

LLM-written SPARQL is parsed, restricted to `SELECT`, stripped of `SERVICE` calls and given a
row limit before it runs. The in-memory rdflib backend cannot enforce query timeouts; the HTTP
backend sets a client timeout. This is cell-line research data, not clinical decision support.

## License

MIT, see [LICENSE](LICENSE).
