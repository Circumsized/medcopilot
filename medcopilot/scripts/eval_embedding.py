"""Chinese dense-retrieval evaluation gate (W4.1).

Ranks every corpus chunk for every query of the synthetic pair set
(``scripts/build_embedding_eval.py``) and reports recall@1/3/5, MRR@5 and the
safety-critical recall@5 (contraindications / interactions).

Honesty rules baked into this script:

* dense retrieval absent (``ENABLE_CHROMA_DENSE=false``, chromadb missing, or
  weights unavailable) ⇒ ``status="skip"`` and exit 0. A degraded BM25 path is
  never reported as a hybrid-retrieval result.
* ``--encoder synthetic`` is a *harness* check (deterministic hash vectors), not
  a model measurement: the report says so in ``warning`` and the gate is
  report-only.
* The thresholds are explicit constants here, never derived from whatever the
  encoder under test happened to score.
* The comparison against the previous English ONNX baseline is
  ``BASELINE_STATUS = "not-evaluated"`` until that baseline is actually measured
  in this environment — no fabricated improvement numbers.

Usage::

    python medcopilot/scripts/eval_embedding.py --encoder synthetic --no-gate
    python medcopilot/scripts/eval_embedding.py            # real dense encoder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPORT_SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "medcopilot" / "datasets" / "eval" / "rag_golden_v1.jsonl"
DEFAULT_DATASET = REPO_ROOT / "medcopilot" / "datasets" / "eval" / "embedding_pairs_v1.jsonl"

# Explicit gate constants (see module docstring: never derived from the run).
THRESHOLDS: dict[str, float] = {
    "recall_at_1": 0.60,
    "recall_at_3": 0.85,
    "recall_at_5": 0.90,
    "mrr_at_5": 0.75,
    "safety_recall_at_5": 0.90,
}
# Relative improvement vs the English ONNX baseline cannot be claimed until that
# baseline is measured here.
BASELINE_STATUS = "not-evaluated"


class SyntheticEncoder:
    """Deterministic hash-based "encoder": a harness fixture, never a measurement.

    Exists so the metric plumbing and the SKIP/gate reporting can be tested
    offline. Similarity between these vectors carries no semantic meaning.
    """

    dim = 64

    def __call__(self, texts: Sequence[str] | str) -> list[list[float]]:
        # Single-item input returns the bare vector (same convention as the
        # production embedding function / ChromaDenseEncoder wrappers).
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = [self._vector(text) for text in items]
        return vectors[0] if single or len(vectors) == 1 else vectors

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[index % len(digest)] / 255.0 for index in range(self.dim)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [value / norm for value in raw]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


def evaluate_pairs(
    pairs: Sequence[dict[str, Any]], ranked: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Deterministic metrics for ``ranked[i]`` answering ``pairs[i]``."""
    if len(ranked) != len(pairs):
        raise ValueError(
            f"ranked list count {len(ranked)} does not match pair count {len(pairs)}; "
            "every pair needs its ranked chunk ids"
        )
    if not pairs:
        raise ValueError("no pairs to evaluate")

    hits = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    safety_total = 0
    safety_hits = 0
    for pair, ranking in zip(pairs, ranked):
        target = pair["chunk_id"]
        position = next((index + 1 for index, chunk_id in enumerate(ranking[:5]) if chunk_id == target), None)
        if position is not None:
            reciprocal += 1.0 / position
            for k in hits:
                if position <= k:
                    hits[k] += 1
        if pair.get("safety_critical"):
            safety_total += 1
            if position is not None:
                safety_hits += 1

    total = len(pairs)
    return {
        "pair_count": total,
        "recall_at_1": hits[1] / total,
        "recall_at_3": hits[3] / total,
        "recall_at_5": hits[5] / total,
        "mrr_at_5": reciprocal / total,
        "safety_pair_count": safety_total,
        "safety_share": safety_total / total,
        "safety_recall_at_5": (safety_hits / safety_total) if safety_total else 0.0,
    }


def apply_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Compare the metrics against the explicit thresholds; never hide the numbers."""
    failures = [
        name
        for name, threshold in THRESHOLDS.items()
        if name in report and float(report[name]) < threshold
    ]
    return {
        "status": "fail" if failures else "pass",
        "failed_metrics": failures,
        "thresholds": dict(THRESHOLDS),
        "relative_improvement_vs_english_onnx": BASELINE_STATUS,
    }


def build_dense_encoder(args: argparse.Namespace):
    """Real dense encoder, or None when dense retrieval is not configured/available."""
    from medcopilot.common.config import config

    if not config.enable_chroma_dense:
        return None
    from medcopilot.rag.vector_store import ChromaDenseEncoder

    return ChromaDenseEncoder.build_encoder(
        persist_directory=config.chroma_persist_directory,
        collection_name=getattr(args, "collection", "med_copilot_eval"),
    )


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") != "meta"]


def _load_chunks(corpus_path: Path) -> list[Any]:
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks: list[Any] = []
    for row in rows:
        if row.get("type") != "corpus":
            continue
        chunks.extend(ClinicalChunker.chunk_drug_label(row["text"], drug_name=row["drug_name"]))
    return chunks


def _rank_all(pairs: Sequence[dict[str, Any]], chunks: Sequence[Any], encoder) -> list[list[str]]:
    texts = [chunk.page_content for chunk in chunks]
    try:
        chunk_vectors = list(encoder(texts))
    except Exception:
        chunk_vectors = [list(encoder(text)) for text in texts]
    chunk_ids = [str(chunk.metadata.get("chunk_id", "")) for chunk in chunks]

    ranked: list[list[str]] = []
    for pair in pairs:
        raw = encoder([pair["query"]])
        query_vector = list(raw[0]) if raw and isinstance(raw[0], (list, tuple)) else list(raw)
        scored = sorted(
            zip(chunk_ids, (_cosine(query_vector, vector) for vector in chunk_vectors)),
            key=lambda item: (-item[1], item[0]),
        )
        ranked.append([chunk_id for chunk_id, _ in scored[:5]])
    return ranked


def breakdown(pairs, ranked) -> dict[str, Any]:
    """细粒度分层:按 style 与 safety_critical 各自给出 recall@5。

    总分达标但某一类问法全丢分的情形,只有分层能暴露——这正是"精细化"
    评测的价值:分数要能指向"哪种问法不可靠"。
    """
    buckets: dict[str, dict[str, list]] = {"by_style": {}, "by_safety": {}}
    for pair, ranking in zip(pairs, ranked):
        target = pair["chunk_id"]
        hit = target in list(ranking[:5])
        for group, key in (
            ("by_style", str(pair.get("style", "unknown"))),
            ("by_safety", bool(pair.get("safety_critical"))),
        ):
            slot = buckets[group].setdefault(key, {"hits": 0, "count": 0})
            slot["count"] += 1
            slot["hits"] += int(hit)
    return {
        group: {
            key: {"count": slot["count"], "recall_at_5": slot["hits"] / slot["count"]}
            for key, slot in sorted(slots.items(), key=lambda item: str(item[0]))
        }
        for group, slots in buckets.items()
    }


def rank_via_pipeline(pairs, chunks, encoder) -> list[list[str]]:
    """L2 管线诊断:走真实 HybridRetriever(BM25+稠密 RRF),而非裸余弦排名。

    评测的是服务实现本身——翻转 ENABLE_CHROMA_DENSE 的证据应来自这里。
    """
    from medcopilot.rag.hybrid_retriever import HybridRetriever

    retriever = HybridRetriever(documents=list(chunks), dense_encoder=encoder)
    ranked: list[list[str]] = []
    for pair in pairs:
        hits = retriever.retrieve_with_scores(pair["query"], top_n=5)
        ranked.append([str(doc.metadata.get("chunk_id", "")) for doc, _ in hits])
    return ranked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chinese dense-retrieval evaluation gate")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--encoder", choices=("auto", "dense", "api", "synthetic"), default="auto")
    parser.add_argument("--pipeline", action="store_true",
                        help="add the L2 diagnostic: real HybridRetriever (RRF) vs BM25-only")
    parser.add_argument("--collection", default="med_copilot_eval")
    parser.add_argument("--no-gate", action="store_true", help="report only; do not fail on thresholds")
    parser.add_argument("--output", default=None, help="optional report path (stdout always gets the JSON)")
    args = parser.parse_args(argv)

    from medcopilot.scripts.build_embedding_eval import MIN_PAIRS

    pairs = _load_pairs(Path(args.dataset))
    if len(pairs) < MIN_PAIRS:
        report = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "status": "fail",
            "reason": f"dataset holds {len(pairs)} pairs, below the required {MIN_PAIRS}",
        }
        _emit(report, args.output)
        return 1

    if args.encoder == "synthetic":
        encoder = SyntheticEncoder()
        warning = (
            "synthetic hash encoder: this harness verifies the metric plumbing and is "
            "not a model measurement"
        )
    else:
        if args.encoder == "api":
            from medcopilot.rag.embedding_api import build_api_encoder

            encoder = build_api_encoder()
            if encoder is None:
                # 显式要求 API 却没配置:这是配置错误,不是"不可用"——SKIP 会把
                # 拼错的密钥静默降级成 BM25 结论。
                print(
                    "FAIL: --encoder api requires EMBEDDING_API_BASE and EMBEDDING_API_KEY",
                    file=sys.stderr,
                )
                return 1
            warning = (
                "hosted embedding API: queries leave this machine; measurement is of the "
                "provider-hosted model, not local weights"
            )
        else:
            encoder = build_dense_encoder(args)
            warning = None
            if encoder is None:
                report = {
                    "report_schema_version": REPORT_SCHEMA_VERSION,
                    "status": "skip",
                    "encoder": args.encoder,
                    "reason": (
                        "dense retrieval unavailable (ENABLE_CHROMA_DENSE=false, chromadb or weights "
                        "missing); BM25-only serving is not a hybrid-retrieval result"
                    ),
                    "pair_count": len(pairs),
                    "git": None,
                }
                _emit(report, args.output)
                print("EMBEDDING EVAL: SKIP (dense retrieval unavailable)", file=sys.stderr)
                return 0

    chunks = _load_chunks(Path(args.corpus))
    ranked = _rank_all(pairs, chunks, encoder)
    metrics = evaluate_pairs(pairs, ranked)
    gate = apply_gate(metrics)
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "status": "report-only" if (args.no_gate or args.encoder == "synthetic") else gate["status"],
        "encoder": args.encoder,
        "encoder_identity": getattr(encoder, "name", lambda: "custom")(),
        "encoder_dim": getattr(encoder, "embedding_dim", None),
        "dataset": str(args.dataset),
        "pair_count": metrics["pair_count"],
        "metrics": metrics,
        # 细粒度分层:总分之外,指出哪类问法在丢分。
        "breakdown": breakdown(pairs, ranked),
        "gate": "report-only" if (args.no_gate or args.encoder == "synthetic") else gate,
        "thresholds": dict(THRESHOLDS),
        "baseline": BASELINE_STATUS,
    }
    if args.pipeline:
        # L2 管线诊断:同一批 pair 在真实 HybridRetriever(RRF 混合)与 BM25 单路
        # 下的对照——为 ENABLE_CHROMA_DENSE 翻转积累服务侧证据。无阈值门。
        hybrid_ranked = rank_via_pipeline(pairs, chunks, encoder)
        bm25_ranked = rank_via_pipeline(pairs, chunks, None)
        report["pipeline"] = {
            "hybrid_rrf": evaluate_pairs(pairs, hybrid_ranked),
            "bm25_only": evaluate_pairs(pairs, bm25_ranked),
            "note": "diagnostic comparison of the serving retriever; no gate applies",
        }
    if warning:
        report["warning"] = warning
    _emit(report, args.output)
    if args.no_gate or args.encoder == "synthetic":
        return 0
    return 0 if gate["status"] == "pass" else 1


def _emit(report: dict[str, Any], output: str | None) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    raise SystemExit(main())
