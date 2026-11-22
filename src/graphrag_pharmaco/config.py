"""Runtime settings, read from environment variables (prefix GRAPHRAG_)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    kg_backend: str = "rdflib"  # "rdflib" (in-memory, from files) or "http" (SPARQL endpoint)
    kg_files: tuple[str, ...] = ()
    sparql_endpoint: str = ""  # e.g. http://localhost:7200/repositories/<repo>
    kg_ns: str = "https://example.org/pharmaco#"
    llm_provider: str = "anthropic"  # "anthropic" or "openai_compat" (local servers, vLLM, ...)
    llm_model: str = ""  # required for real LLM use; deliberately has no default
    llm_base_url: str = ""  # for openai_compat, e.g. http://localhost:8000/v1
    max_rows: int = 200
    sparql_timeout_s: float = 30.0
    cache_size: int = 512
    cache_ttl_s: float = 600.0

    @classmethod
    def from_env(cls) -> Settings:
        def get(name: str, default: str = "") -> str:
            return os.environ.get(f"GRAPHRAG_{name}", default)

        files = tuple(p for p in get("KG_FILES").split(",") if p.strip())
        return cls(
            kg_backend=get("KG_BACKEND", cls.kg_backend),
            kg_files=files,
            sparql_endpoint=get("SPARQL_ENDPOINT"),
            kg_ns=get("KG_NS", cls.kg_ns),
            llm_provider=get("LLM_PROVIDER", cls.llm_provider),
            llm_model=get("LLM_MODEL"),
            llm_base_url=get("LLM_BASE_URL"),
            max_rows=int(get("MAX_ROWS", str(cls.max_rows))),
            sparql_timeout_s=float(get("SPARQL_TIMEOUT_S", str(cls.sparql_timeout_s))),
            cache_size=int(get("CACHE_SIZE", str(cls.cache_size))),
            cache_ttl_s=float(get("CACHE_TTL_S", str(cls.cache_ttl_s))),
        )
