"""The KG contract this repo expects. See docs/KG_CONTRACT.md.

The namespace is configurable; class and property local names are fixed so the KG repo
and this repo can evolve independently against one documented interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

ENTITY_CLASSES = ("Drug", "CellLine", "Gene", "Pathway", "Tissue")


@dataclass(frozen=True)
class Schema:
    ns: str = "https://example.org/pharmaco#"

    def uri(self, local: str) -> str:
        return f"{self.ns}{local}"

    def local(self, uri: str) -> str:
        if uri.startswith(self.ns):
            return uri[len(self.ns):]
        return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]

    @property
    def prefix(self) -> str:
        return (
            f"PREFIX ph: <{self.ns}>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
        )

    def with_prefixes(self, query: str) -> str:
        """Prepend any of ph/rdfs/rdf/xsd that the query uses but does not declare."""
        declared = {m.lower() for m in re.findall(r"(?im)^\s*PREFIX\s+(\w*):", query)}
        missing = [
            line for line in self.prefix.splitlines()
            if line.split()[1].rstrip(":").lower() not in declared
        ]
        return "\n".join(missing + [query]) if missing else query

    def description(self) -> str:
        """Schema summary placed into text-to-SPARQL prompts."""
        return (
            f"{self.prefix}\n"
            "Classes: ph:Drug, ph:CellLine, ph:Gene, ph:Pathway, ph:Tissue,\n"
            "  ph:Response (a MEASURED drug response), ph:PredictedResponse (a MODEL PREDICTION).\n"
            "Every entity has an rdfs:label.\n"
            "Properties:\n"
            "  ph:Drug      ph:targets        ph:Gene\n"
            "  ph:Drug      ph:targetPathway  ph:Pathway\n"
            "  ph:CellLine  ph:tissue         ph:Tissue\n"
            "  ph:Response           ph:ofDrug ph:Drug ; ph:ofCellLine ph:CellLine ;\n"
            "                        ph:auc xsd:double   (LOWER AUC = MORE sensitive)\n"
            "  ph:PredictedResponse  ph:ofDrug ph:Drug ; ph:ofCellLine ph:CellLine ;\n"
            "                        ph:score xsd:double (HIGHER score = predicted MORE sensitive)\n"
            "Nothing else is stored: no doses, approval status, trial data or literature.\n"
        )
