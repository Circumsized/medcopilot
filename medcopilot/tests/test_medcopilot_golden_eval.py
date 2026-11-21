"""Slice 4 contract: versioned RAG golden dataset, deterministic harness, failure closed loop.

The golden evaluation must be runnable offline (lexical BM25 path + mock-safe
routing/abstention paths only), must gate on a versioned dataset artifact, and
must emit replayable failure samples instead of silently passing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "medcopilot" / "datasets" / "eval" / "rag_golden_v1.jsonl"


def _write_jsonl(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


CORPUS_IBU = (
    "【药品名称】布洛芬缓释胶囊\n"
    "【适应症】用于缓解轻至中度疼痛如头痛、关节痛。\n"
    "【用法用量】口服。一次1粒，一日2次。\n"
    "【禁忌】活动期消化道溃疡及出血患者禁用。"
)
CORPUS_AMOX = (
    "【药品名称】阿莫西林胶囊\n"
    "【适应症】用于敏感菌所致的呼吸道感染、泌尿道感染。\n"
    "【禁忌】青霉素过敏患者禁用。"
)


def _synthetic_dataset(tmp_path: Path, cases, corpora=None) -> Path:
    records = [{"type": "meta", "dataset": "rag_golden", "version": 1}]
    for corpus in (corpora or [
        {"type": "corpus", "corpus_id": "ibu-v1", "drug_name": "布洛芬缓释胶囊", "text": CORPUS_IBU},
        {"type": "corpus", "corpus_id": "amox-v1", "drug_name": "阿莫西林胶囊", "text": CORPUS_AMOX},
    ]):
        records.append(corpus)
    records.extend(cases)
    return _write_jsonl(tmp_path / "golden.jsonl", records)


def _load(path: Path):
    from medcopilot.evaluation.golden_eval import load_golden_dataset

    return load_golden_dataset(path)


class TestDatasetValidation:
    def test_valid_dataset_loads(self, tmp_path):
        path = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "ibu-contraindication", "kind": "retrieval",
            "question": "胃溃疡可以吃布洛芬缓释胶囊吗",
            "expect": {"hit_drug": "布洛芬缓释胶囊", "hit_sections": ["禁忌"]},
        }])
        dataset = _load(path)
        assert dataset.version == 1
        assert {c["corpus_id"] for c in dataset.corpora} == {"ibu-v1", "amox-v1"}
        assert len(dataset.cases) == 1

    def test_missing_meta_header_rejected(self, tmp_path):
        path = _write_jsonl(tmp_path / "g.jsonl", [
            {"type": "corpus", "corpus_id": "a", "drug_name": "药", "text": "【药品名称】药\n【禁忌】禁"},
            {"type": "case", "case_id": "c1", "kind": "retrieval", "question": "q",
             "expect": {"hit_drug": "药", "hit_sections": ["禁忌"]}},
        ])
        with pytest.raises(Exception, match="meta"):
            _load(path)

    def test_unknown_kind_rejected_with_case_id(self, tmp_path):
        path = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "weird-1", "kind": "sentiment",
            "question": "q", "expect": {},
        }])
        with pytest.raises(Exception, match="weird-1"):
            _load(path)

    def test_retrieval_case_requires_expectation_fields(self, tmp_path):
        path = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "no-expect", "kind": "retrieval",
            "question": "q", "expect": {"hit_sections": ["禁忌"]},
        }])
        with pytest.raises(Exception, match="no-expect"):
            _load(path)

    def test_malformed_line_reports_line_number(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text(
            '{"type": "meta", "dataset": "rag_golden", "version": 1}\n' "{not json}\n",
            encoding="utf-8",
        )
        with pytest.raises(Exception, match="line 2"):
            _load(path)

    def test_empty_dataset_rejected(self, tmp_path):
        # 空金标集永远不能给出 PASS —— 评测门不允许被掏空。
        path = _synthetic_dataset(tmp_path, [])
        with pytest.raises(Exception, match="case"):
            _load(path)

    def test_duplicate_case_or_corpus_ids_rejected(self, tmp_path):
        path = _synthetic_dataset(tmp_path, [
            {"type": "case", "case_id": "dup", "kind": "abstention", "question": "q",
             "expect": {"abstained": True}},
            {"type": "case", "case_id": "dup", "kind": "abstention", "question": "q2",
             "expect": {"abstained": True}},
        ])
        with pytest.raises(Exception, match="dup"):
            _load(path)


class TestRetrievalHarness:
    def test_expected_section_hit_passes(self, tmp_path):
        from medcopilot.evaluation.golden_eval import run_golden

        path = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "ibu-contraindication", "kind": "retrieval",
            "question": "胃溃疡可以吃布洛芬缓释胶囊吗",
            "expect": {"hit_drug": "布洛芬缓释胶囊", "hit_sections": ["禁忌"], "top_k": 3},
        }])
        report = run_golden(_load(path))
        assert not report.failed
        assert report.retrieval_hit_rate == 1.0

    def test_wrong_section_expectation_fails_with_actual(self, tmp_path):
        from medcopilot.evaluation.golden_eval import run_golden

        path = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "impossible", "kind": "retrieval",
            "question": "布洛芬缓释胶囊多少钱一盒",
            "expect": {"hit_drug": "阿莫西林胶囊", "hit_sections": ["儿童用药"], "top_k": 1},
        }])
        report = run_golden(_load(path))
        assert len(report.failed) == 1
        failure = report.failed[0]
        assert failure.case_id == "impossible"
        # 失败载荷必须包含可复现现场：检索到了什么。
        assert "hits" in failure.actual


class TestRoutingAndAbstentionHarness:
    def test_emergency_case_passes(self, tmp_path):
        from medcopilot.evaluation.golden_eval import run_golden

        path = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "emergency-1", "kind": "routing",
            "question": "胸口压榨性疼痛喘不上气，是不是心梗了？",
            "expect": {"intent": "emergency"},
        }])
        assert not run_golden(_load(path)).failed

    def test_non_medical_refusal_case_passes(self, tmp_path):
        from medcopilot.evaluation.golden_eval import run_golden

        path = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "non-medical-1", "kind": "routing",
            "question": "帮我写一篇关于春天的作文",
            "expect": {"intent": "non_medical"},
        }])
        report = run_golden(_load(path))
        assert not report.failed
        assert report.results[0].actual["refused"] is True

    def test_abstention_for_absent_drug_passes(self, tmp_path):
        from medcopilot.evaluation.golden_eval import run_golden

        path = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "abstain-1", "kind": "abstention",
            "question": "枸橼酸西地那非片第一次吃多少剂量合适？",
            "expect": {"abstained": True},
        }])
        assert not run_golden(_load(path)).failed


class TestFailureClosedLoop:
    def test_failure_samples_written_as_jsonl(self, tmp_path):
        from medcopilot.evaluation.golden_eval import run_golden, write_failure_samples

        path = _synthetic_dataset(tmp_path, [
            {"type": "case", "case_id": "ok-case", "kind": "retrieval",
             "question": "青霉素过敏能吃阿莫西林胶囊吗",
             "expect": {"hit_drug": "阿莫西林胶囊", "hit_sections": ["禁忌"]}},
            {"type": "case", "case_id": "bad-case", "kind": "retrieval",
             "question": "阿莫西林胶囊什么价格",
             "expect": {"hit_drug": "阿莫西林胶囊", "hit_sections": ["老年用药"], "top_k": 1}},
        ])
        out = tmp_path / "out" / "failures.jsonl"
        report = run_golden(_load(path))
        written = write_failure_samples(report, out)
        assert written is not None and written.exists()
        lines = [json.loads(x) for x in written.read_text(encoding="utf-8").splitlines() if x]
        assert [x["case_id"] for x in lines] == ["bad-case"]
        for key in ("dataset_version", "kind", "question", "expected", "actual"):
            assert key in lines[0]

    def test_no_failures_writes_nothing(self, tmp_path):
        from medcopilot.evaluation.golden_eval import run_golden, write_failure_samples

        path = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "ok-case", "kind": "retrieval",
            "question": "胃溃疡可以吃布洛芬缓释胶囊吗",
            "expect": {"hit_drug": "布洛芬缓释胶囊", "hit_sections": ["禁忌"]},
        }])
        out = tmp_path / "failures.jsonl"
        assert write_failure_samples(run_golden(_load(path)), out) is None
        assert not out.exists()


class TestGoldenDatasetArtifact:
    """守护真实金标工件本身：版本、覆盖度与标签一致性。"""

    def test_dataset_file_exists_and_versioned(self):
        assert GOLDEN_PATH.exists(), f"missing golden dataset {GOLDEN_PATH}"
        dataset = _load(GOLDEN_PATH)
        assert dataset.version == 1
        assert len(dataset.cases) >= 12

    def test_kind_coverage(self):
        dataset = _load(GOLDEN_PATH)
        kinds = [c["kind"] for c in dataset.cases]
        assert kinds.count("retrieval") >= 8
        assert kinds.count("routing") >= 3
        assert kinds.count("abstention") >= 2

    def test_labels_reference_real_corpus_sections(self):
        from medcopilot.rag.clinical_chunker import ClinicalChunker

        dataset = _load(GOLDEN_PATH)
        drugs = {c["drug_name"] for c in dataset.corpora}
        sections_by_drug = {
            corpus["drug_name"]: {
                doc.metadata.get("section")
                for doc in ClinicalChunker.chunk_drug_label(corpus["text"], drug_name=corpus["drug_name"])
            }
            for corpus in dataset.corpora
        }
        for case in dataset.cases:
            if case["kind"] != "retrieval":
                continue
            hit_drug = case["expect"]["hit_drug"]
            assert hit_drug in drugs, f"{case['case_id']}: phantom drug label"
            for section in case["expect"]["hit_sections"]:
                assert section in sections_by_drug[hit_drug], (
                    f"{case['case_id']}: {hit_drug} has no section {section}"
                )


class TestAcceptanceGate:
    def test_real_golden_dataset_fully_passes(self):
        from medcopilot.evaluation.golden_eval import run_golden

        report = run_golden(_load(GOLDEN_PATH))
        failures = [
            f"{r.case_id}: expected={r.expected} actual={r.actual}" for r in report.failed
        ]
        assert not report.failed, "golden eval failures:\n" + "\n".join(failures)
        assert report.retrieval_hit_rate == 1.0


class TestCli:
    def _run_cli(self, args, extra_env=None):
        env = dict(os.environ)
        env["MOCK_LLM"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT)
        env.update(extra_env or {})
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "medcopilot" / "scripts" / "eval_golden.py"), *args],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=300,
        )

    def test_cli_passes_on_real_dataset_exit_zero(self, tmp_path):
        proc = self._run_cli(["--failures-out", str(tmp_path / "f.jsonl")])
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert not (tmp_path / "f.jsonl").exists()

    def test_cli_failure_writes_samples_and_exits_one(self, tmp_path, monkeypatch):
        bad = _synthetic_dataset(tmp_path, [{
            "type": "case", "case_id": "bad-cli", "kind": "retrieval",
            "question": "阿莫西林胶囊什么价格",
            "expect": {"hit_drug": "阿莫西林胶囊", "hit_sections": ["老年用药"], "top_k": 1},
        }])
        out = tmp_path / "out" / "f.jsonl"
        proc = self._run_cli(["--golden", str(bad), "--failures-out", str(out)])
        assert proc.returncode == 1, proc.stdout + proc.stderr
        assert out.exists()
        assert "bad-cli" in out.read_text(encoding="utf-8")

    def test_cli_broken_dataset_exits_two(self, tmp_path):
        broken = tmp_path / "broken.jsonl"
        broken.write_text('{"type": "meta"}\n', encoding="utf-8")
        proc = self._run_cli(["--golden", str(broken)])
        assert proc.returncode == 2
