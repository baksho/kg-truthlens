"""FastAPI service. Run: uvicorn graphrag_pharmaco.api:app  (reads GRAPHRAG_* env vars)."""

from __future__ import annotations

import time
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graphrag_pharmaco.cache import TTLCache
from graphrag_pharmaco.config import Settings
from graphrag_pharmaco.llm import build_llm
from graphrag_pharmaco.system import STRATEGIES, System, build_system


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strategy: str = "hybrid"


class EvidenceOut(BaseModel):
    id: str
    text: str
    status: str


class AskResponse(BaseModel):
    answer: list[str]
    citations: list[str]
    abstain: bool
    note: str
    evidence: list[EvidenceOut]
    sparql: str | None
    strategy: str
    latency_s: float
    cached: bool = False


def create_app(system: System | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="graphrag-pharmaco")
    answers: TTLCache[AskResponse] = TTLCache(settings.cache_size, settings.cache_ttl_s)
    state: dict[str, System] = {}
    if system is not None:
        state["system"] = system

    def get_system() -> System:
        if "system" not in state:  # lazy: the app can start before the triplestore is up
            state["system"] = build_system(settings, llm=build_llm(settings))
        return state["system"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "strategies": list(STRATEGIES), "answer_cache_size": len(answers)}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        if req.strategy not in STRATEGIES:
            raise HTTPException(422, f"strategy must be one of {STRATEGIES}")
        key = f"{req.strategy}|{' '.join(req.question.lower().split())}"
        hit = answers.get(key)
        if hit is not None:
            return hit.model_copy(update={"cached": True})
        try:
            started = time.perf_counter()
            retrieval, answer = get_system().ask(req.question, req.strategy)
        except RuntimeError as exc:  # misconfiguration (no LLM, no KG)
            raise HTTPException(503, str(exc)) from exc
        response = AskResponse(
            answer=answer.answer, citations=answer.citations, abstain=answer.abstain, note=answer.note,
            evidence=[EvidenceOut(id=e.id, text=e.text, status=e.status) for e in retrieval.evidence],
            sparql=retrieval.sparql, strategy=req.strategy, latency_s=time.perf_counter() - started,
        )
        if answer.parse_ok:
            answers.put(key, response)
        return response

    return app


@lru_cache(maxsize=1)
def _default_app() -> FastAPI:
    return create_app()


def __getattr__(name: str):  # `uvicorn graphrag_pharmaco.api:app` builds the app on first access
    if name == "app":
        return _default_app()
    raise AttributeError(name)
