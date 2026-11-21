"""W3.4 chunker v2 契约:结构边界、剂量保护、reranker 可见长度对齐。"""

from __future__ import annotations

import re

from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.reranker import Reranker


def test_dose_and_regimen_not_split():
    text = "患者用药方案：一次0.375g，一日2次，饭后服用。" + ("说明文字。" * 150)
    parts = ClinicalChunker._split_long(text)
    joined = "".join(parts)
    assert "0.375g" in joined
    assert "一日2次" in joined
    # 每个切片不得以小数/剂量数字悬空结尾(重叠连接除外)
    assert not any(re.search(r"(?:\d+\.\d+|一日\d+)$", p) for p in parts[:-1])


def test_chunker_prefers_newline_and_sentence_boundaries():
    text = ("表头|剂量|频次\n" + "药品A|0.5g|一日2次\n") * 100
    parts = ClinicalChunker._split_long(text)
    assert all(len(p) <= ClinicalChunker.HARD_MAX_CHARS for p in parts)
    # 每个切点附近保留行边界(重叠会让 chunk 尾部本身不一定落在边界上)。
    assert all("\n" in p[-80:] for p in parts[:-1])


def test_chunker_v2_limits_long_chunks_for_reranker():
    parts = ClinicalChunker._split_long("头痛。" * 1000)
    assert max(map(len, parts)) <= ClinicalChunker.HARD_MAX_CHARS
    assert ClinicalChunker.CHUNK_MAX_CHARS == 480
    assert ClinicalChunker.CHUNK_OVERLAP == 60


def test_reranker_max_length_is_configurable():
    reranker = Reranker(enabled=False, max_length=1024)
    assert reranker.max_length == 1024
    assert Reranker(enabled=False).max_length == 512
