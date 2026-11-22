"""Turn raw triples into readable evidence lines.

Measured (ph:Response) and predicted (ph:PredictedResponse) nodes are reified in the KG;
here they are collapsed into one line each, with an explicit 'measured' / 'predicted' status
so the generator can never confuse the two.
"""

from __future__ import annotations

from graphrag_pharmaco.kg.backend import SparqlBackend, batches, is_uri
from graphrag_pharmaco.kg.labels import LabelStore
from graphrag_pharmaco.kg.schema import Schema


class Verbalizer:
    def __init__(self, backend: SparqlBackend, schema: Schema, labels: LabelStore):
        self.backend = backend
        self.schema = schema
        self.labels = labels

    def triple_text(self, s: str, p: str, o: str) -> str:
        return f"{self.labels.render(s)} --{self.schema.local(p)}--> {self.labels.render(o)}"

    def response_nodes(self, uris: list[str]) -> dict[str, tuple[str, str]]:
        """Map response-node URI -> (status, text). Unknown nodes are omitted."""
        ph = self.schema.uri
        out: dict[str, tuple[str, str]] = {}
        for chunk in batches([u for u in uris if is_uri(u)], 100):
            values = " ".join(f"<{u}>" for u in chunk)
            rows = self.backend.select(
                f"SELECT ?r ?type ?d ?c ?auc ?score WHERE {{ VALUES ?r {{ {values} }} "
                f"?r a ?type ; <{ph('ofDrug')}> ?d ; <{ph('ofCellLine')}> ?c . "
                f"OPTIONAL {{ ?r <{ph('auc')}> ?auc }} OPTIONAL {{ ?r <{ph('score')}> ?score }} "
                f"FILTER(?type IN (<{ph('Response')}>, <{ph('PredictedResponse')}>)) }}"
            )
            self.labels.prefetch([v for row in rows for v in (row["d"], row["c"])])
            for row in rows:
                drug, cell = self.labels.label(row["d"]), self.labels.label(row["c"])
                if row["type"] == ph("PredictedResponse"):
                    text = f"PREDICTED (model output, not a measurement): {drug} on {cell}, score={row.get('score', '?')}"
                    out[row["r"]] = ("predicted", text)
                else:
                    text = f"MEASURED: {drug} on {cell}, AUC={row.get('auc', '?')}"
                    out[row["r"]] = ("measured", text)
        return out
