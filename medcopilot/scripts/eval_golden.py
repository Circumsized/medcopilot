"""Golden-set acceptance gate for the Med-Copilot RAG pipeline (Slice 4).

Deterministic, offline behavioral gate over the versioned dataset artifact:
lexical retrieval hit placement, triage routing and evidence-free abstention.
Exit codes: 0 all cases pass; 1 at least one failure (samples written for the
closed loop); 2 the dataset artifact itself is invalid.

Answer faithfulness under a real LLM is NOT measured here (see
scripts/eval_rag.py for the conservative extractive proxy). A PASS from this
gate means behavioral contracts hold — never claim clinical correctness.

Usage:
    MOCK_LLM=1 python medcopilot/scripts/eval_golden.py
    python medcopilot/scripts/eval_golden.py --golden medcopilot/datasets/eval/rag_golden_v1.jsonl \
        --failures-out reports/golden_failures.jsonl --report-out reports/golden_report.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medcopilot.evaluation.golden_eval import (
    GoldenDatasetError,
    load_golden_dataset,
    run_golden,
    write_failure_samples,
)

DEFAULT_GOLDEN = Path(__file__).resolve().parents[1] / "datasets" / "eval" / "rag_golden_v1.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Golden-set behavioral gate (retrieval/routing/abstention)")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="versioned golden JSONL artifact")
    parser.add_argument("--failures-out", default="reports/golden_failures.jsonl", help="failure-sample closed loop")
    parser.add_argument("--report-out", default=None, help="optional JSON metrics report for trend tracking")
    args = parser.parse_args()

    try:
        dataset = load_golden_dataset(args.golden)
    except GoldenDatasetError as exc:
        print(f"GOLDEN DATASET INVALID: {exc}")
        sys.exit(2)

    report = run_golden(dataset)
    written = write_failure_samples(report, args.failures_out)
    if args.report_out:
        out = Path(args.report_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.summary(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report.summary(), ensure_ascii=False, indent=2))
    if written is not None:
        print(f"failure samples written to {written} (replay inputs for the next iteration)")
    print("PASS" if not report.failed else f"FAIL ({len(report.failed)}/{len(report.results)} cases)")
    sys.exit(0 if not report.failed else 1)


if __name__ == "__main__":
    main()
