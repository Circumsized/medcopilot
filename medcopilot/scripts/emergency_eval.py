"""急救红旗验收集门禁(emergency_eval)。

退出码:0 = 三项指标全部达标;1 = 任一指标不达标;2 = 数据集非法。
指标(阈值取自数据集 meta.gates):
- red_flag_recall@vignettes        = 触发数 / red_flag 条数      (必须 1.00)
- benign_fp_rate                   = 误触发数 / benign_trap 条数  (必须 ≤ 0.05)
- history_demotion_accuracy        = 未触发数 / history_mention 条数(必须 1.00)

用法::

    MOCK_LLM=1 python medcopilot/scripts/emergency_eval.py [--dataset PATH] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medcopilot.workflow.safety_guard import MedicalSafetyGuard  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "eval" / "emergency_vignettes_v1.jsonl"
_VALID_TYPES = {"meta", "red_flag", "benign_trap", "history_mention"}


def load_cases(path: Path) -> tuple[dict, list[dict]]:
    cases: list[dict] = []
    meta: dict = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict) or record.get("type") not in _VALID_TYPES:
                raise ValueError(f"{path}:{lineno} has invalid/missing type {record.get('type')!r}")
            if record["type"] == "meta":
                meta = record
            else:
                if not record.get("case_id") or not record.get("query"):
                    raise ValueError(f"{path}:{lineno} missing case_id/query")
                cases.append(record)
    if not meta:
        raise ValueError(f"{path}: missing meta record")
    return meta, cases


def evaluate(path: Path) -> tuple[dict, int]:
    meta, cases = load_cases(path)
    gates = meta.get("gates") or {}
    recall_target = float(gates.get("red_flag_recall", 1.0))
    fp_max = float(gates.get("benign_fp_rate_max", 0.05))
    demotion_target = float(gates.get("history_demotion_accuracy", 1.0))

    red_flags = [c for c in cases if c["type"] == "red_flag"]
    benign = [c for c in cases if c["type"] == "benign_trap"]
    history = [c for c in cases if c["type"] == "history_mention"]
    if not red_flags or not benign:
        raise ValueError("dataset must contain red_flag and benign_trap cases")

    fired = [c for c in red_flags if MedicalSafetyGuard.check_emergency(c["query"])[0]]
    false_positives = [c for c in benign if MedicalSafetyGuard.check_emergency(c["query"])[0]]
    demoted = [c for c in history if not MedicalSafetyGuard.check_emergency(c["query"])[0]]

    recall = len(fired) / len(red_flags)
    fp_rate = len(false_positives) / len(benign)
    demotion = (len(demoted) / len(history)) if history else 1.0

    report = {
        "dataset": str(path),
        "dataset_version": meta.get("version"),
        "red_flag_total": len(red_flags),
        "red_flag_fired": len(fired),
        "red_flag_recall": round(recall, 4),
        "benign_total": len(benign),
        "benign_false_positives": [c["case_id"] for c in false_positives],
        "benign_fp_rate": round(fp_rate, 4),
        "history_total": len(history),
        "history_demotion_accuracy": round(demotion, 4),
        "gates": {
            "red_flag_recall": recall_target,
            "benign_fp_rate_max": fp_max,
            "history_demotion_accuracy": demotion_target,
        },
        "passed": (
            recall >= recall_target
            and fp_rate <= fp_max
            and demotion >= demotion_target
        ),
    }
    missed = [c["case_id"] for c in red_flags if c not in fired]
    if missed:
        report["missed_red_flags"] = missed
    return report, 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="emergency red-flag vignette gate")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="path to the vignette dataset")
    parser.add_argument("--json", dest="json_out", default=None, help="optional path for the JSON report")
    args = parser.parse_args(argv)

    path = Path(args.dataset)
    try:
        report, code = evaluate(path)
    except (OSError, ValueError) as exc:
        print(f"FAIL: invalid emergency vignette dataset: {exc}", file=sys.stderr)
        return 2

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if code == 0:
        print(
            f"PASS: emergency v{report['dataset_version']} red_flag_recall="
            f"{report['red_flag_recall']} benign_fp_rate={report['benign_fp_rate']} "
            f"history_demotion_accuracy={report['history_demotion_accuracy']}"
        )
    else:
        print("FAIL: emergency vignette gate not met:", file=sys.stderr)
        if report.get("missed_red_flags"):
            print(f"  - missed red flags: {report['missed_red_flags']}", file=sys.stderr)
        if report["benign_false_positives"]:
            print(f"  - benign false positives: {report['benign_false_positives']}", file=sys.stderr)
        if report["history_demotion_accuracy"] < report["gates"]["history_demotion_accuracy"]:
            print("  - history demotion below target", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
