# KG contract

This repo consumes an RDF export from the separate KG repo. This is the interface between
them. It is a proposal: if the KG repo's schema differs, either adapt the KG export or change
`src/graphrag_pharmaco/kg/schema.py` (the only place class/property names are defined; the
namespace is set with `GRAPHRAG_KG_NS`).

| Term | Kind | Meaning |
|---|---|---|
| `ph:Drug`, `ph:CellLine`, `ph:Gene`, `ph:Pathway`, `ph:Tissue` | classes | Entities. Each needs an `rdfs:label`. |
| `ph:targets` | Drug → Gene | Putative drug target. |
| `ph:targetPathway` | Drug → Pathway | Target pathway annotation. |
| `ph:tissue` | CellLine → Tissue | Tissue of origin. |
| `ph:Response` | class | A **measured** drug/cell-line response (reified). |
| `ph:ofDrug`, `ph:ofCellLine` | Response/PredictedResponse → entity | The pair the value belongs to. |
| `ph:auc` | Response → xsd:double | Lower AUC = more sensitive. Confirm this convention for your data source. |
| `ph:PredictedResponse` | class | A **model prediction** (e.g. link-prediction output), never a measurement. |
| `ph:score` | PredictedResponse → xsd:double | Higher = predicted more sensitive. |

Rules this repo relies on:

1. Measured and predicted values are different classes, never the same node type with a flag.
   The answer generator is told which is which, and the eval checks that it keeps them apart.
2. The graph holds nothing about doses, approval status or trials. The benchmark's
   "unanswerable" questions depend on that.
3. Real data only. The benchmark gold answers are computed from this graph with SPARQL.

Export format: Turtle (`.ttl`) or N-Triples. Load with `GRAPHRAG_KG_FILES` (in-memory rdflib, fine
for development and small graphs) or into GraphDB with `scripts/load_graphdb.py`.
