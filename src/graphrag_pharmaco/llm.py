"""LLM adapters. Anything with `complete(system, user) -> str` works."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import httpx

from graphrag_pharmaco.config import Settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM:
    """Requires `pip install .[anthropic]` and ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic  # imported lazily so the base install stays light

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


class OpenAICompatLLM:
    """Any server exposing the OpenAI-style /chat/completions route (vLLM, llama.cpp, Ollama /v1)."""

    def __init__(self, model: str, base_url: str, api_key: str = "", timeout_s: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout_s)

    def complete(self, system: str, user: str) -> str:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class ScriptedLLM:
    """Deterministic stand-in for tests: `fn(system, user)` returns the reply."""

    def __init__(self, fn: Callable[[str, str], str]):
        self.fn = fn
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.fn(system, user)


def build_llm(settings: Settings) -> LLM:
    import os

    if not settings.llm_model:
        raise RuntimeError("Set GRAPHRAG_LLM_MODEL (no default is baked in; check your provider's model list).")
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.llm_model)
    if settings.llm_provider == "openai_compat":
        if not settings.llm_base_url:
            raise RuntimeError("Set GRAPHRAG_LLM_BASE_URL for the openai_compat provider.")
        return OpenAICompatLLM(settings.llm_model, settings.llm_base_url, os.environ.get("GRAPHRAG_LLM_API_KEY", ""))
    raise RuntimeError(f"Unknown GRAPHRAG_LLM_PROVIDER: {settings.llm_provider}")
