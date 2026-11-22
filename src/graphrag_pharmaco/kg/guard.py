"""Validation of LLM-written SPARQL before it touches the store."""

from __future__ import annotations

import re

from rdflib.plugins.sparql import prepareQuery


class QueryRejected(ValueError):
    """Raised when a query is syntactically invalid or not allowed."""


_LIMIT_TAIL = re.compile(r"\bLIMIT\s+(\d+)(\s+OFFSET\s+\d+)?\s*$", re.IGNORECASE)
_SERVICE = re.compile(r"\bSERVICE\b", re.IGNORECASE)


def guard(query: str, max_rows: int = 200) -> str:
    """Return a safe version of `query`, or raise QueryRejected.

    Rules: must parse as SPARQL, must be a SELECT (no updates), no federated SERVICE calls,
    and a row limit of at most `max_rows` is always enforced.
    """
    text = query.strip().rstrip(";").strip()
    if not text:
        raise QueryRejected("empty query")
    if _SERVICE.search(text):
        raise QueryRejected("SERVICE (federated) queries are not allowed")
    try:
        prepared = prepareQuery(text)
    except Exception as exc:  # rdflib raises several parser exception types
        raise QueryRejected(f"invalid SPARQL: {exc}") from exc
    if prepared.algebra.name != "SelectQuery":
        raise QueryRejected("only SELECT queries are allowed")
    match = _LIMIT_TAIL.search(text)
    if match is None:
        return f"{text}\nLIMIT {max_rows}"
    if int(match.group(1)) > max_rows:
        return text[: match.start(1)] + str(max_rows) + text[match.end(1):]
    return text


_FENCE = re.compile(r"```(?:sparql|SPARQL)?\s*(.*?)```", re.DOTALL)


def extract_sparql(llm_output: str) -> str:
    """Pull the query out of a fenced code block if there is one, else use the raw text."""
    match = _FENCE.search(llm_output)
    return (match.group(1) if match else llm_output).strip()
