"""Versioned golden evaluation for the Med-Copilot RAG pipeline.

Scope discipline: this harness gates *behavioral contracts* that are fully
deterministic offline — lexical retrieval hit placement, triage routing
(emergency / non-medical) and evidence-free abstention. Answer faithfulness
under a real LLM is a separate, non-gating report (see ``medcopilot/scripts/eval_rag.py``
for the conservative extractive proxy); this module never fabricates a PASS
from disclaimer or mock text.

Dataset artifact: JSONL under ``medcopilot/datasets/eval/`` — the file IS the versioned
contract (``medcopilot/datasets/`` is git-tracked; ``data/`` is the ignored runtime dir).
Record types: one ``meta`` header, then ``corpus`` records (simplified teaching
labels) and ``case`` records (question + expectations). Labels describe the
clinically correct expected outcome, not whatever the current code does.

Failure closed loop: every failing case is written as one JSONL sample with
question, expectation and the retrieval/routing actuals, so failures are
replayable inputs for the next iteration instead of opaque red CI lines.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from medcopilot.common.logger import logger

GOLDEN_DATASET_VERSION = 1
DEFAULT_TOP_K = 3
_VALID_KINDS = {"retrieval", "routing", "abstention"}
_VALID_INTENTS = {"emergency", "non_medical", "medical"}


class GoldenDatasetError(ValueError):
    """Raised when the golden dataset artifact itself violates its schema."""


@dataclass
class GoldenDataset:
    version: int
    corpora: list[dict]
    cases: list[dict]


@dataclass
class CaseResult:
    case_id: str
    kind: str
    question: str
    passed: bool
    expected: dict
    actual: dict


@dataclass
class GoldenReport:
    dataset_version: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def failed(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    @property
    def retrieval_hit_rate(self) -> float:
        retrieval = [r for r in self.results if r.kind == "retrieval"]
        if not retrieval:
            return 0.0
        return sum(1 for r in retrieval if r.passed) / len(retrieval)

    def summary(self) -> dict:
        return {
            "dataset_version": self.dataset_version,
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.passed),
            "failed": len(self.failed),
            "retrieval_hit_rate": round(self.retrieval_hit_rate, 3),
            "cases": [
                {"case_id": r.case_id, "kind": r.kind, "passed": r.passed} for r in self.results
            ],
        }


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    """Parse and validate the versioned JSONL artifact. Never silently permissive."""
    path = Path(path)
    if not path.exists():
        raise GoldenDatasetError(f"golden dataset not found: {path}")
    meta_version: int | None = None
    corpora: list[dict] = []
    cases: list[dict] = []
    seen_case_ids: set[str] = set()
    seen_corpus_ids: set[str] = set()

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    records: list[tuple[int, dict]] = []
    for lineno, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenDatasetError(f"invalid JSON at line {lineno}: {exc}") from exc
        if not isinstance(record, dict):
            raise GoldenDatasetError(f"line {lineno}: record must be a JSON object")
        records.append((lineno, record))

    if not records:
        raise GoldenDatasetError("dataset contains no records")
    first_lineno, first = records[0]
    if first.get("type") != "meta":
        raise GoldenDatasetError(f"line {first_lineno}: first record must be a meta header")
    meta_version = first.get("version")
    if not isinstance(meta_version, int) or meta_version < 1:
        raise GoldenDatasetError("meta header requires a positive integer version")
    if meta_version > GOLDEN_DATASET_VERSION:
        raise GoldenDatasetError(
            f"dataset version {meta_version} requires a newer loader (max {GOLDEN_DATASET_VERSION})"
        )

    for lineno, record in records[1:]:
        record_type = record.get("type")
        if record_type == "corpus":
            corpus_id = record.get("corpus_id")
            drug_name = record.get("drug_name")
            text = record.get("text")
            if not (isinstance(corpus_id, str) and corpus_id and isinstance(drug_name, str)
                    and drug_name and isinstance(text, str) and text):
                raise GoldenDatasetError(
                    f"line {lineno}: corpus record requires corpus_id, drug_name and text"
                )
            if corpus_id in seen_corpus_ids:
                raise GoldenDatasetError(f"duplicate corpus_id '{corpus_id}' at line {lineno}")
            seen_corpus_ids.add(corpus_id)
            corpora.append(record)
        elif record_type == "case":
            case_id = record.get("case_id")
            kind = record.get("kind")
            expect = record.get("expect")
            if not isinstance(case_id, str) or not case_id:
                raise GoldenDatasetError(f"line {lineno}: case record requires case_id")
            if case_id in seen_case_ids:
                raise GoldenDatasetError(f"duplicate case_id '{case_id}' at line {lineno}")
            if kind not in _VALID_KINDS:
                raise GoldenDatasetError(f"case {case_id}: unknown kind '{kind}'")
            if not isinstance(expect, dict):
                raise GoldenDatasetError(f"case {case_id}: requires an expect object")
            if kind == "retrieval" and (
                not isinstance(expect.get("hit_drug"), str) or not expect.get("hit_drug")
                or not isinstance(expect.get("hit_sections"), list) or not expect["hit_sections"]
            ):
                raise GoldenDatasetError(
                    f"case {case_id}: retrieval case requires hit_drug and hit_sections"
                )
            if kind == "routing" and expect.get("intent") not in _VALID_INTENTS:
                raise GoldenDatasetError(f"case {case_id}: routing expect.intent must be one of {sorted(_VALID_INTENTS)}")
            if kind == "abstention" and not isinstance(expect.get("abstained"), bool):
                raise GoldenDatasetError(f"case {case_id}: abstention case requires boolean expect.abstained")
            if not isinstance(record.get("question"), str) or not record["question"]:
                raise GoldenDatasetError(f"case {case_id}: requires a non-empty question")
            seen_case_ids.add(case_id)
            cases.append(record)
        else:
            raise GoldenDatasetError(f"line {lineno}: unknown record type '{record_type}'")

    if not corpora:
        raise GoldenDatasetError("dataset contains no corpus records")
    if not cases:
        raise GoldenDatasetError("dataset contains no cases — the gate may never be emptied")
    return GoldenDataset(version=meta_version, corpora=corpora, cases=cases)


def run_golden(dataset: GoldenDataset) -> GoldenReport:
    """Execute every case against the live pipeline and return per-case results.

    Retrieval cases exercise the lexical (BM25) retrieval path directly, which is
    deterministic and dependency-free. Routing/abstention cases execute the full
    agent graph; those routes are designed to terminate before any LLM call, so
    results are identical under mock and real providers for correctly routed
    cases. Never use this harness to measure answer faithfulness.
    """
    from medcopilot.agents.triage import TriageAgent
    from medcopilot.rag.clinical_chunker import ClinicalChunker
    from medcopilot.rag.hybrid_retriever import HybridRetriever
    from medcopilot.rag.reranker import Reranker
    from medcopilot.workflow.agent_graph import AgentGraph

    chunks = []
    for corpus in dataset.corpora:
        docs = ClinicalChunker.chunk_drug_label(corpus["text"], drug_name=corpus["drug_name"])
        for doc in docs:
            doc.metadata["corpus_id"] = corpus["corpus_id"]
            chunks.append(doc)
    retriever = HybridRetriever(documents=chunks)
    graph = AgentGraph(retriever=retriever, reranker=Reranker(enabled=False))

    report = GoldenReport(dataset_version=dataset.version)
    for case in dataset.cases:
        kind = case["kind"]
        question = case["question"]
        expect = case["expect"]
        if kind == "retrieval":
            top_k = int(expect.get("top_k", DEFAULT_TOP_K))
            hits = retriever.retrieve(question, top_n=max(top_k, 1))
            hit_drug = expect["hit_drug"]
            hit_sections = set(expect["hit_sections"])
            actual = {
                "hits": [
                    {
                        "drug_name": doc.metadata.get("drug_name"),
                        "section": doc.metadata.get("section"),
                    }
                    for doc in hits
                ]
            }
            passed = any(
                h["drug_name"] == hit_drug and h["section"] in hit_sections for h in actual["hits"]
            )
        elif kind == "routing":
            result = graph.run(question)
            expected_intent = expect["intent"]
            actual = {
                "intent": result["intent"],
                "is_emergency": bool(result.get("is_emergency")),
                "abstained": bool(result.get("abstained")),
                # run() exposes intent; refusal is exactly the non_medical route.
                "refused": result["intent"] == TriageAgent.NON_MEDICAL,
            }
            passed = result["intent"] == expected_intent
            if expected_intent == "emergency":
                passed = passed and actual["is_emergency"]
        else:  # abstention
            result = graph.run(question)
            actual = {
                "abstained": bool(result.get("abstained")),
                "intent": result["intent"],
            }
            passed = actual["abstained"] is bool(expect["abstained"])
        report.results.append(
            CaseResult(
                case_id=case["case_id"], kind=kind, question=question,
                passed=bool(passed), expected=dict(expect), actual=actual,
            )
        )
        if not passed:
            logger.warning("Golden case {} failed: expected {} actual {}", case["case_id"], expect, actual)
    return report


def write_failure_samples(report: GoldenReport, out_path: str | Path) -> Path | None:
    """Persist replayable failure samples; returns None when nothing failed."""
    failures = report.failed
    if not failures:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "dataset_version": report.dataset_version,
                "case_id": r.case_id,
                "kind": r.kind,
                "question": r.question,
                "expected": r.expected,
                "actual": r.actual,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for r in failures
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
