"""Deterministic builder for the Chinese retrieval evaluation set (W4.1).

The pairs are generated from the repository's own teaching labels
(``datasets/eval/rag_golden_v1.jsonl``), so they are **synthetic**: they measure
whether an encoder can rank the right *label section* above the other sections
of the same corpus. They are not real clinical queries and must never be
described as such.

Why generate instead of hand-write:

* reproducibility — the committed dataset is byte-identical to this script's
  output (``--check`` enforces that in CI);
* coverage of the hard cases the design calls for: colloquial phrasing, a weak
  section hint (the section name is absent), comorbidity framing and
  combination framing;
* the safety-critical share is a hard constraint, not a happy accident.

Usage::

    python medcopilot/scripts/build_embedding_eval.py            # regenerate
    python medcopilot/scripts/build_embedding_eval.py --check     # verify
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATASET_SCHEMA_VERSION = 1
DATASET_NAME = "embedding_pairs"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "medcopilot" / "datasets" / "eval" / "rag_golden_v1.jsonl"
DEFAULT_TARGET = REPO_ROOT / "medcopilot" / "datasets" / "eval" / "embedding_pairs_v1.jsonl"

# Sections whose retrieval failure can change a clinical decision (contra-
# indications and interactions). Everything else is informative-only.
SAFETY_SECTIONS = frozenset({"禁忌", "药物相互作用"})
# Query styles. The first three apply to every chunk; the rest only to safety
# chunks, which is what lifts the safety share above the floor.
STYLES = ("canonical", "colloquial", "combination", "weak-hint", "risk", "symptom")
NON_SAFETY_STYLES = ("canonical", "colloquial", "symptom")

MIN_PAIRS = 50
MIN_SAFETY_SHARE = 0.40

# Query templates per style. ``{drug}`` / ``{section}`` / ``{keyword}`` are the
# only substitutions; ``weak-hint`` deliberately omits the section name so the
# encoder cannot win on the section word alone.
TEMPLATES: dict[str, str] = {
    "canonical": "{drug}的{section}是什么？",
    "colloquial": "{drug}这个药，{section}方面要留意啥？",
    "symptom": "{drug}{keyword}，这种情况用药要注意什么？",
    "combination": "正在吃别的药，还能用{drug}吗？{keyword}",
    "weak-hint": "{drug}，{keyword}，有什么必须提醒我的？",
    "risk": "{drug}{keyword}会有什么风险或后果？",
}

_SECTION_HEADER = re.compile(r"【([^】]+)】")
_CLAUSE_SPLIT = re.compile(r"[，。；、,;]")


def load_corpus(path: str | Path = DEFAULT_CORPUS) -> list[dict[str, Any]]:
    """Corpus rows (``type == "corpus"``) from the golden dataset, in file order."""
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row.get("type") == "corpus"]


def _section_bodies(text: str) -> dict[str, str]:
    """Map section name -> body text, preserving the label's own order."""
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        header = _SECTION_HEADER.match(line.strip())
        if header:
            current = header.group(1)
            remainder = _SECTION_HEADER.sub("", line).strip()
            bodies.setdefault(current, [])
            if remainder:
                bodies[current].append(remainder)
        elif current is not None and line.strip():
            bodies[current].append(line.strip())
    return {name: "\n".join(parts) for name, parts in bodies.items()}


def extract_keyword(body: str) -> str:
    """First clause of a section body — deterministic, always non-empty.

    The clause is drawn from the label itself, so no medical claim is invented
    by this script; it only rephrases the label into a question.
    """
    for clause in _CLAUSE_SPLIT.split(body):
        cleaned = clause.strip(" 　\t\n。，；、")
        if len(cleaned) >= 2:
            return cleaned[:14]
    return body.strip()[:14] or "用药"


def build_pairs(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic query–chunk pairs for every label section of every drug."""
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    pairs: list[dict[str, Any]] = []
    for row in corpus:
        corpus_id = row["corpus_id"]
        drug_name = row["drug_name"]
        bodies = _section_bodies(row["text"])
        chunks = ClinicalChunker.chunk_drug_label(row["text"], drug_name=drug_name)
        for index, chunk in enumerate(chunks):
            section = str(chunk.metadata.get("section", ""))
            if section not in bodies:
                continue
            safety = section in SAFETY_SECTIONS
            keyword = extract_keyword(bodies[section])
            styles = STYLES if safety else NON_SAFETY_STYLES
            for style in styles:
                pairs.append(
                    {
                        "pair_id": f"{corpus_id}-{index:02d}-{style}",
                        "corpus_id": corpus_id,
                        "drug_name": drug_name,
                        "section": section,
                        "chunk_id": str(chunk.metadata.get("chunk_id", "")),
                        "query": TEMPLATES[style].format(drug=drug_name, section=section, keyword=keyword),
                        "style": style,
                        "safety_critical": safety,
                        "synthetic": True,
                    }
                )
    return pairs


def build_dataset(corpus: list[dict[str, Any]], target: str | Path = DEFAULT_TARGET) -> Path:
    """Validate the constraints, then write the dataset; never write a bad one."""
    pairs = build_pairs(corpus)
    if len(pairs) < MIN_PAIRS:
        raise ValueError(f"pair count {len(pairs)} is below the required {MIN_PAIRS}")
    safety = sum(1 for pair in pairs if pair["safety_critical"])
    share = safety / len(pairs)
    if share < MIN_SAFETY_SHARE:
        raise ValueError(
            f"safety share {share:.3f} is below the required {MIN_SAFETY_SHARE}"
        )
    if len({pair["pair_id"] for pair in pairs}) != len(pairs):
        raise ValueError("pair ids are not unique")

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "type": "meta",
        "dataset": DATASET_NAME,
        "version": DATASET_SCHEMA_VERSION,
        "synthetic": True,
        "note": (
            "Synthetic query-chunk pairs generated from the repository's teaching labels; "
            "these are not real clinical queries and the set does not establish clinical accuracy."
        ),
        "pair_count": len(pairs),
        "safety_pair_count": safety,
        "safety_share": round(share, 4),
    }
    lines = [json.dumps(meta, ensure_ascii=False)] + [
        json.dumps(pair, ensure_ascii=False) for pair in pairs
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the synthetic Chinese retrieval eval set")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed dataset matches this script instead of writing it",
    )
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    target = Path(args.target)
    if args.check:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            probe = Path(tmp_dir) / target.name
            try:
                build_dataset(corpus, probe)
            except ValueError as exc:
                print(f"FAIL: {exc}", file=sys.stderr)
                return 1
            matches = probe.read_text(encoding="utf-8") == target.read_text(encoding="utf-8")
        print("EMBEDDING EVAL SET: up to date" if matches else "FAIL: committed dataset is stale; regenerate")
        return 0 if matches else 1

    try:
        target = build_dataset(corpus, args.target)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    pairs = build_pairs(corpus)
    safety = sum(1 for pair in pairs if pair["safety_critical"])
    print(
        f"EMBEDDING EVAL SET: {len(pairs)} pairs "
        f"(safety {safety}, share {safety / len(pairs):.1%}) -> {target}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
