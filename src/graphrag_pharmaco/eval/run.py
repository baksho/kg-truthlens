"""CLI: generate a benchmark from the KG and compare retrieval strategies.

  python -m graphrag_pharmaco.eval.run --kg data/kg.ttl --retrieval-only --out runs/r0
  GRAPHRAG_LLM_MODEL=... python -m graphrag_pharmaco.eval.run --kg data/kg.ttl --out runs/r1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

from graphrag_pharmaco.config import Settings
from graphrag_pharmaco.eval.metrics import ItemScore, mean, score_answer, score_retrieval
from graphrag_pharmaco.eval.questions import Question, generate_questions
from graphrag_pharmaco.llm import LLM, build_llm
from graphrag_pharmaco.system import STRATEGIES, System, build_system

METRICS = [f.name for f in dataclasses.fields(ItemScore)]


def run_eval(system: System, questions: list[Question], strategies: list[str], retrieval_only: bool) -> list[dict]:
    records = []
    for strategy in strategies:
        for q in questions:
            retriever = system.retriever(strategy)
            retrieval = retriever.retrieve(q.question)
            if retrieval_only or system.llm is None:
                score = score_retrieval(q, retrieval)
                answer = None
            else:
                from graphrag_pharmaco.generation import generate

                answer = generate(q.question, retrieval, system.llm)
                score = dataclasses.asdict(score_answer(q, retrieval, answer))
            records.append({
                "strategy": strategy, "question": q.to_dict(),
                "sparql": retrieval.sparql, "error": retrieval.error, "latency_s": retrieval.latency_s,
                "n_evidence": len(retrieval.evidence),
                "answer": dataclasses.asdict(answer) if answer else None, "score": score,
            })
    return records


def summarize(records: list[dict]) -> dict[str, dict[str, dict[str, float | None]]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["strategy"], r["question"]["kind"])].append(r)
        groups[(r["strategy"], "ALL")].append(r)
    table: dict[str, dict[str, dict[str, float | None]]] = defaultdict(dict)
    for (strategy, kind), rs in sorted(groups.items()):
        row = {m: mean([x["score"].get(m) for x in rs]) for m in METRICS}
        row["latency_s"] = mean([x["latency_s"] for x in rs])
        row["n"] = len(rs)
        table[strategy][kind] = row
    return table


def to_markdown(table: dict[str, dict[str, dict[str, float | None]]], cols: list[str]) -> str:
    def fmt(v):
        return "–" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    lines = ["| strategy | kind | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 3)]
    for strategy, kinds in table.items():
        for kind, row in kinds.items():
            lines.append(f"| {strategy} | {kind} | {row['n']} | " + " | ".join(fmt(row.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None, llm: LLM | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg", nargs="+", required=True, help="RDF files (the KG repo's export)")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--n-per-template", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrieval-only", action="store_true", help="skip generation; no LLM needed except for text2sparql/hybrid")
    ap.add_argument("--out", default="runs/latest")
    args = ap.parse_args(argv)

    settings = dataclasses.replace(Settings.from_env(), kg_backend="rdflib", kg_files=tuple(args.kg))
    strategies = [s for s in args.strategies.split(",") if s]
    needs_llm = any(s in ("text2sparql", "hybrid") for s in strategies) or not args.retrieval_only
    system = build_system(settings, llm=llm or (build_llm(settings) if needs_llm else None))
    questions = generate_questions(system.backend, system.schema, args.n_per_template, args.seed)
    records = run_eval(system, questions, strategies, args.retrieval_only)
    table = summarize(records)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "questions.json").write_text(json.dumps([q.to_dict() for q in questions], indent=2))
    (out / "records.json").write_text(json.dumps(records, indent=2))
    cols = ["retrieval_recall", "linking_recall"] if args.retrieval_only else [
        "f1", "recall", "precision", "abstain_correct", "over_abstain", "citation_validity",
        "unsupported_rate", "retrieval_recall", "sparql_valid", "latency_s"]
    md = to_markdown(table, cols)
    (out / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
