"""W4.1: 中文检索评测集的约束 + 确定性指标（全离线，不下载权重）。

评测集是从仓库自带的教学说明书语料生成的**合成** query–chunk 对，不是真实
临床查询；因此：

* 数量与安全占比有硬约束（>=50 对、禁忌/相互作用 >=40%）；
* 指标是确定性的闭式计算（recall@1/3/5、MRR@5、安全召回@5），可被钉死；
* dense 缺席（默认 `ENABLE_CHROMA_DENSE=false`、或缺 chromadb/权重）→ 打印
  SKIP 且退出码 0，绝不把降级路径粉饰成"混合检索达标"；
* 真实模型阈值是显式常量；没有英文 ONNX 基线实测之前，相对提升项如实标
  not-evaluated，不编造任何真实模型收益。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medcopilot.scripts import build_embedding_eval as builder
from medcopilot.scripts import eval_embedding as evaluator

REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRS_PATH = REPO_ROOT / "medcopilot" / "datasets" / "eval" / "embedding_pairs_v1.jsonl"
GOLDEN_PATH = REPO_ROOT / "medcopilot" / "datasets" / "eval" / "rag_golden_v1.jsonl"

REQUIRED_KEYS = {
    "pair_id",
    "corpus_id",
    "drug_name",
    "section",
    "chunk_id",
    "query",
    "style",
    "safety_critical",
    "synthetic",
}


# ------------------------------------------------------------------ 数据集约束


def _rows():
    return [json.loads(line) for line in PAIRS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pairs(rows):
    return [row for row in rows if row.get("type") != "meta"]


def test_dataset_exists_and_is_versioned():
    rows = _rows()
    assert rows, "评测集不能为空"
    meta = rows[0]
    assert meta["type"] == "meta"
    assert meta["dataset"] == "embedding_pairs"
    assert meta["version"] == 1
    assert meta["synthetic"] is True
    assert "synthetic" in meta["note"].lower()


def test_dataset_has_at_least_fifty_pairs():
    assert len(_pairs(_rows())) >= builder.MIN_PAIRS


def test_safety_critical_share_is_at_least_forty_percent():
    pairs = _pairs(_rows())
    safety = [p for p in pairs if p["safety_critical"]]
    assert len(safety) / len(pairs) >= builder.MIN_SAFETY_SHARE


def test_every_pair_is_well_formed_and_uniquely_identified():
    rows = _rows()
    pairs = _pairs(rows)
    for row in pairs:
        missing = REQUIRED_KEYS - set(row)
        assert not missing, f"{row.get('pair_id')} missing {missing}"
        assert row["synthetic"] is True
        assert row["query"].strip()
        assert row["style"] in builder.STYLES
        assert isinstance(row["safety_critical"], bool)
        assert row["chunk_id"]
    assert len({row["pair_id"] for row in pairs}) == len(pairs)


def test_safety_labels_only_cover_safety_sections():
    for row in _pairs(_rows()):
        if row["safety_critical"]:
            assert row["section"] in builder.SAFETY_SECTIONS, row
        else:
            assert row["section"] not in builder.SAFETY_SECTIONS, row


def test_every_pair_points_at_a_real_chunk_in_the_golden_corpus():
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    corpus = [
        json.loads(line)
        for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = {row["corpus_id"]: row for row in corpus if row.get("type") == "corpus"}
    known_ids = set()
    for row in labels.values():
        for chunk in ClinicalChunker.chunk_drug_label(row["text"], drug_name=row["drug_name"]):
            known_ids.add(chunk.metadata["chunk_id"])
    pairs = _pairs(_rows())
    assert {row["corpus_id"] for row in pairs} <= set(labels)
    unknown = [row["chunk_id"] for row in pairs if row["chunk_id"] not in known_ids]
    assert not unknown, f"评测集引用了不存在的 chunk: {unknown}"


# ------------------------------------------------------------------ 生成确定性


def test_build_is_deterministic_and_matches_the_committed_dataset():
    rows = _rows()
    pairs = _pairs(rows)
    rebuilt = builder.build_pairs(builder.load_corpus(GOLDEN_PATH))
    assert rebuilt == pairs, "提交的评测集必须与脚本输出逐字节一致（可复现）"


def test_build_refuses_to_write_a_dataset_that_violates_constraints(tmp_path, monkeypatch):
    """约束不达标必须是显式失败，不能悄悄写出一份不达标的评测集。"""

    class _LyingBuilder:
        MIN_PAIRS = 999

    monkeypatch.setattr(builder, "MIN_PAIRS", 999)
    with pytest.raises(ValueError, match="pair count"):
        builder.build_dataset([], tmp_path / "out.jsonl")
    assert not (tmp_path / "out.jsonl").exists()
    assert _LyingBuilder.MIN_PAIRS == 999


def test_build_writes_a_meta_header_and_valid_rows(tmp_path):
    target = tmp_path / "pairs.jsonl"
    builder.build_dataset(builder.load_corpus(GOLDEN_PATH), target)
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["type"] == "meta"
    assert builder.MIN_PAIRS <= len(rows) - 1
    # 再跑一次：同样输入 → 同样输出。
    again = tmp_path / "pairs2.jsonl"
    builder.build_dataset(builder.load_corpus(GOLDEN_PATH), again)
    assert again.read_text(encoding="utf-8") == target.read_text(encoding="utf-8")


# ------------------------------------------------------------------ 确定性指标


def _ranked(*ids):
    return list(ids)


def test_metrics_are_deterministic_and_match_a_closed_form():
    pairs = [
        {"pair_id": "p1", "chunk_id": "a", "safety_critical": True},
        {"pair_id": "p2", "chunk_id": "b", "safety_critical": False},
        {"pair_id": "p3", "chunk_id": "c", "safety_critical": True},
    ]
    ranked = [_ranked("a", "x", "y"), _ranked("a", "b", "c"), _ranked("x", "y", "z")]
    report = evaluator.evaluate_pairs(pairs, ranked)

    assert report["pair_count"] == 3
    assert report["recall_at_1"] == pytest.approx(1 / 3)
    assert report["recall_at_3"] == pytest.approx(2 / 3)
    assert report["recall_at_5"] == pytest.approx(2 / 3)
    assert report["mrr_at_5"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert report["safety_pair_count"] == 2
    assert report["safety_share"] == pytest.approx(2 / 3)
    assert report["safety_recall_at_5"] == pytest.approx(0.5)


def test_metrics_are_stable_to_ordering_of_pairs():
    pairs = [
        {"pair_id": "p1", "chunk_id": "a", "safety_critical": True},
        {"pair_id": "p2", "chunk_id": "b", "safety_critical": False},
    ]
    first = evaluator.evaluate_pairs(pairs, [_ranked("a", "z"), _ranked("b", "z")])
    second = evaluator.evaluate_pairs(
        list(reversed(pairs)), list(reversed([_ranked("a", "z"), _ranked("b", "z")]))
    )
    assert first == second


def test_missing_ranked_list_is_not_silently_scored():
    with pytest.raises(ValueError, match="ranked"):
        evaluator.evaluate_pairs([{"pair_id": "p", "chunk_id": "a", "safety_critical": False}], [])


def test_thresholds_are_explicit_constants():
    """阈值必须是显式常量，不能由被测 enchancoder 的表现推导。"""
    assert evaluator.THRESHOLDS["recall_at_3"] == 0.85
    assert evaluator.THRESHOLDS["safety_recall_at_5"] == 0.9
    assert evaluator.BASELINE_STATUS == "not-evaluated"


def test_gate_reports_failure_without_hiding_the_numbers():
    report = evaluator.evaluate_pairs(
        [{"pair_id": "p", "chunk_id": "a", "safety_critical": False}], [_ranked("z", "y")]
    )
    gate = evaluator.apply_gate(report)
    assert gate["status"] == "fail"
    assert gate["thresholds"] == evaluator.THRESHOLDS
    assert gate["relative_improvement_vs_english_onnx"] == "not-evaluated"


# ------------------------------------------------------------------ SKIP 语义


def test_dense_absent_means_skip_and_exit_zero(tmp_path, monkeypatch):
    """默认（dense 关）必须 SKIP + exit 0，而不是把降级粉饰成达标。"""
    from medcopilot.common.config import config

    monkeypatch.setattr(config, "enable_chroma_dense", False)
    monkeypatch.setattr(evaluator, "build_dense_encoder", lambda args: None)
    monkeypatch.setattr(
        "sys.argv",
        ["eval_embedding.py", "--corpus", str(GOLDEN_PATH), "--dataset", str(PAIRS_PATH)],
    )
    code = evaluator.main(["--corpus", str(GOLDEN_PATH), "--dataset", str(PAIRS_PATH)])
    assert code == 0


def test_report_labels_a_synthetic_harness_as_such(tmp_path, monkeypatch, capsys):
    """--encoder synthetic 是夹具而非模型测量：报告必须显式标注。"""
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_embedding.py", "--corpus", str(GOLDEN_PATH), "--dataset", str(PAIRS_PATH),
            "--encoder", "synthetic",
        ],
    )
    code = evaluator.main(
        [
            "--corpus", str(GOLDEN_PATH), "--dataset", str(PAIRS_PATH), "--encoder", "synthetic",
            "--no-gate",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["encoder"] == "synthetic"
    assert out["gate"] == "report-only"
    assert "not a model measurement" in out["warning"]
    assert out["metrics"]["pair_count"] >= builder.MIN_PAIRS


def test_synthetic_encoder_is_deterministic():
    first = evaluator.SyntheticEncoder()(["布洛芬禁忌"])
    second = evaluator.SyntheticEncoder()(["布洛芬禁忌"])
    assert first == second
    assert len(first) == evaluator.SyntheticEncoder.dim


# ------------------------------------------------- API 通道与细粒度报告


def test_breakdown_reports_per_style_and_per_safety():
    from medcopilot.scripts import eval_embedding as evaluator

    pairs = [
        {"pair_id": "p1", "chunk_id": "a", "style": "canonical", "safety_critical": True},
        {"pair_id": "p2", "chunk_id": "b", "style": "canonical", "safety_critical": True},
        {"pair_id": "p3", "chunk_id": "c", "style": "weak-hint", "safety_critical": False},
    ]
    ranked = [_ranked("a", "x"), _ranked("b", "z"), _ranked("q", "z")]
    breakdown = evaluator.breakdown(pairs, ranked)
    assert breakdown["by_style"]["canonical"] == {"count": 2, "recall_at_5": 1.0}
    assert breakdown["by_style"]["weak-hint"] == {"count": 1, "recall_at_5": 0.0}
    assert breakdown["by_safety"][True] == {"count": 2, "recall_at_5": 1.0}
    assert breakdown["by_safety"][False] == {"count": 1, "recall_at_5": 0.0}


def test_pipeline_ranking_uses_the_real_retriever(monkeypatch):
    """L2 管线诊断走真实 HybridRetriever(稠密=注入编码器),可离线打桩。"""
    from medcopilot.rag.clinical_chunker import Document
    from medcopilot.scripts import eval_embedding as evaluator

    chunks = [Document("头痛资料用于临床咨询。", {"chunk_id": "a"}),
              Document("感冒说明资料。", {"chunk_id": "b"})]
    pairs = [{"pair_id": "p1", "chunk_id": "a", "query": "头痛如何处理",
              "style": "canonical", "safety_critical": False}]
    ranked = evaluator.rank_via_pipeline(pairs, chunks, evaluator.SyntheticEncoder())
    assert len(ranked) == len(pairs)
    assert all(isinstance(ids, list) for ids in ranked)


def test_explicit_api_encoder_without_configuration_fails_loud(monkeypatch, capsys):
    from medcopilot.common.config import config
    from medcopilot.scripts import eval_embedding as evaluator

    monkeypatch.setattr(config, "embedding_api_base", "")
    monkeypatch.setattr(config, "embedding_api_key", "")
    code = evaluator.main(["--dataset", str(PAIRS_PATH), "--corpus", str(GOLDEN_PATH),
                           "--encoder", "api", "--no-gate"])
    assert code == 1
    assert "EMBEDDING_API" in capsys.readouterr().err


# ------------------------------------------------- BM25-hard 改写探针集


def test_paraphrase_probe_is_wellformed_and_points_at_real_chunks():
    """探针集:改写/去药名查询,专测词法匹配不上的问法;必须钉在真实 chunk 上。"""
    probe_path = REPO_ROOT / "medcopilot" / "datasets" / "eval" / "paraphrase_probe_v1.jsonl"
    rows = [json.loads(line) for line in probe_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [row for row in rows if row.get("type") != "meta"]
    assert rows[0]["type"] == "meta" and rows[0]["synthetic"] is True
    assert len(pairs) >= 50, "探针集规模须过 eval 的 MIN_PAIRS 下限"
    styles = {"paraphrase", "no-name", "colloquial", "control"}
    for pair in pairs:
        missing = REQUIRED_KEYS - set(pair)
        assert not missing, f"{pair.get('pair_id')} missing {missing}"
        assert pair["style"] in styles, pair
        assert pair["query"].strip()
    assert len({pair["pair_id"] for pair in pairs}) == len(pairs)
    # 与主集同一约束:chunk_id 必须是金标语料真实切出来的。
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    corpus = [json.loads(line) for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    known_ids = {
        chunk.metadata["chunk_id"]
        for row in corpus if row.get("type") == "corpus"
        for chunk in ClinicalChunker.chunk_drug_label(row["text"], drug_name=row["drug_name"])
    }
    unknown = [pair["chunk_id"] for pair in pairs if pair["chunk_id"] not in known_ids]
    assert not unknown, f"探针集引用了不存在的 chunk: {unknown}"
    # 安全标注与主集同义:禁忌/相互作用才算安全对。
    for pair in pairs:
        if pair["safety_critical"]:
            assert pair["section"] in {"禁忌", "药物相互作用"}, pair
        else:
            assert pair["section"] not in {"禁忌", "药物相互作用"}, pair
