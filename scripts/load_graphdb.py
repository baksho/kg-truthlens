"""Load an RDF file into an EXISTING repository of a SPARQL/RDF4J-style server (e.g. GraphDB).

  python scripts/load_graphdb.py http://localhost:7200/repositories/pharmaco data/kg.ttl

Uses the RDF4J REST route `POST <repo>/statements`, which GraphDB documents support for. Not
tested here (no running GraphDB in the authoring environment): verify against your version.
Create the repository first (GraphDB Workbench -> Setup -> Repositories).
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

CONTENT_TYPES = {".ttl": "text/turtle", ".nt": "application/n-triples", ".rdf": "application/rdf+xml"}


def main(repo_url: str, path: str) -> None:
    p = Path(path)
    ctype = CONTENT_TYPES.get(p.suffix.lower())
    if ctype is None:
        sys.exit(f"unsupported extension {p.suffix}; use one of {sorted(CONTENT_TYPES)}")
    r = httpx.post(f"{repo_url.rstrip('/')}/statements", content=p.read_bytes(),
                   headers={"Content-Type": ctype}, timeout=600)
    r.raise_for_status()
    print(f"loaded {p} -> {repo_url} (HTTP {r.status_code})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
