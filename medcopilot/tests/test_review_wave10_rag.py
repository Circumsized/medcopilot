"""Wave 10: 未提交改动专家团评审(expert_review_uncommitted_20261009)的 RAG 域修复契约。

1. C4 boot 期换模型 pin_verified 假绿:生产 encoder boot 时模型名/函数名可得、
   维度尚未实测(dim=None),身份比较必须逐字段独立,不靠 dimension gating 跳过
   model/name;完全不暴露身份的 encoder 不得假绿。
2. chunker overlap 起点落剂量 span:下一块不得以"次。。"式残缺给药片段开头;
   保证有进展(块首严格前进)、不漏文本(相邻块重叠或相接)、不无限循环。
   测的是**实际边界**(块首/块尾偏移),不是"某块包含 span"——重叠让后者永真。
3. reranker 稳定 tie:``docs.index`` 按值比较 + O(n) 每次,值相等的重复文档会抢到
   首个下标插到中间文档之前;改枚举下标后同分必须保持输入顺序。
   (embedding API payload 结构契约不在本文件:真实适配器是
   ``rag/embedding_api.py``,其回归在 test_review_wave10_embedding.py 与
   test_embedding_api.py;vector_store 里的同名副本已按最终 review 删除。)
"""

from __future__ import annotations

import pytest

from medcopilot.rag.clinical_chunker import ClinicalChunker, Document
from medcopilot.rag.corpus_store import CorpusStore
from medcopilot.rag.index_manager import IndexManager
from medcopilot.rag.reranker import Reranker

LABEL = "【药品名称】wave10药品\n【用法用量】口服。一次1片，一日3次。\n【禁忌】过敏者禁用。\n"


# ------------------------------------------------------------- 1: boot 身份(C4)


class _FakeEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def name(self) -> str:
        return "med-bge-m3"


class _IdentityEncoder:
    """声明嵌入身份的 encoder 替身;``dim=None`` 即生产 boot 形态(未实测)。"""

    def __init__(self, collection_name: str, *, dim: int | None = 1024, model: str = "BAAI/bge-m3"):
        self.collection_name = collection_name
        self.docs: list = []
        self.embedding_dim = dim
        self.embedding_function = _FakeEmbeddingFunction(model)

    def index_documents(self, documents) -> int:
        self.docs = list(documents)
        return len(self.docs)

    def count(self) -> int:
        return len(self.docs)

    def indexed_ids(self) -> list[str]:
        return [str(d.metadata.get("chunk_id", "")) for d in self.docs]

    def __call__(self, text):
        return [0.0, 0.0]


class _NoIdentityEncoder:
    """完全不暴露嵌入身份、但 count 与 manifest 钉死数量一致的 encoder。"""

    def __init__(self, collection_name: str, *, count: int) -> None:
        self.collection_name = collection_name
        self._count = count

    def index_documents(self, documents) -> int:
        return len(list(documents))

    def count(self) -> int:
        return self._count

    def __call__(self, text):
        return [0.0, 0.0]


def _manager(tmp_path, encoder_class=_IdentityEncoder) -> IndexManager:
    store = CorpusStore(path=tmp_path / "corpus.jsonl")
    store.upsert_drug("wave10药品", ClinicalChunker.chunk_drug_label(LABEL, drug_name="wave10药品"))

    class _Factory:
        def __call__(self, name: str):
            return encoder_class(name)

    return IndexManager(
        corpus_store=store,
        encoder_factory=_Factory(),
        manifest_path=tmp_path / "index_manifest.json",
        base_collection="med_copilot",
    )


def test_bind_active_compares_model_when_dim_is_unmeasured_at_boot(tmp_path):
    """C4 核心回归:boot 期 dim=None 不得把 model 比较一起跳过(换模型假绿)。"""
    manager = _manager(tmp_path)
    manifest = manager.rebuild()
    assert manifest["embedding_verified"] is True

    swapped = _IdentityEncoder(manifest["collection"], dim=None, model="other/model")
    manager.bind_active(swapped, manifest["collection"])
    assert manager.pin_verified is False, "模型名不符必须判未验证,与维度是否实测无关"


def test_bind_active_rejects_function_name_mismatch_when_dim_is_unmeasured(tmp_path):
    manager = _manager(tmp_path)
    manifest = manager.rebuild()
    encoder = _IdentityEncoder(manifest["collection"], dim=None)
    encoder.embedding_function.name = lambda: "other-fn"
    manager.bind_active(encoder, manifest["collection"])
    assert manager.pin_verified is False


def test_bind_active_accepts_matching_model_with_unmeasured_dim(tmp_path):
    """同一模型 + boot 期未实测维度仍然是合法 pin:不得误报红。"""
    manager = _manager(tmp_path)
    manifest = manager.rebuild()
    encoder = _IdentityEncoder(manifest["collection"], dim=None)
    # boot 形态:新 encoder 打开同一(持久化)collection,装着 manifest 钉住的内容。
    encoder.index_documents(manager.active_encoder.docs)
    manager.bind_active(encoder, manifest["collection"])
    assert manager.pin_verified is True


def test_bind_active_rejects_an_encoder_with_no_identity_even_when_count_matches(tmp_path):
    """无身份不能假绿:数量一致不构成嵌入空间同一性的证据。"""
    manager = _manager(tmp_path)
    manifest = manager.rebuild()
    encoder = _NoIdentityEncoder(manifest["collection"], count=manifest["corpus_chunk_count"])
    manager.bind_active(encoder, manifest["collection"])
    assert manager.pin_verified is False


# ------------------------------------------- 2: chunker overlap 起点剂量保护


def _safe_filler(length: int) -> str:
    """无边界字符、无数字/剂量单位的非周期填充(步长 7,最小周期 2000 > 用例长度)。

    非周期是刚需:``find`` 定位分片偏移时,周期填充会让同一字符串出现在多个
    位置,边界断言随之失真。
    """
    return "".join(chr(0x4E00 + (index * 7) % 2000) for index in range(length))


def _part_offsets(text: str, parts: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    search_from = 0
    for part in parts:
        found = text.find(part, search_from)
        assert found >= 0, "分片必须是原文的连续子串"
        offsets.append((found, found + len(part)))
        search_from = found + 1
    return offsets


def _spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in ClinicalChunker.DOSE_PROTECT_PATTERN.finditer(text)]


def _assert_boundaries_are_clean(text: str, parts: list[str]) -> tuple[list[int], list[int]]:
    """块首/块尾都不得严格落在剂量 span 内;相邻块重叠或相接;块首严格前进。"""
    offsets = _part_offsets(text, parts)
    starts = [start for start, _ in offsets]
    ends = [end for _, end in offsets]
    spans = _spans(text)
    assert spans, "用例必须包含剂量 span"
    assert not any(s < start < e for start in starts[1:] for s, e in spans), (
        f"分块起点落在剂量 span 中间: {starts}, spans={spans}"
    )
    assert not any(s < end < e for end in ends[:-1] for s, e in spans), (
        f"分块切点落在剂量 span 中间: {ends}, spans={spans}"
    )
    assert offsets[0][0] == 0 and offsets[-1][1] == len(text), "首块须始于文首、末块须终于文尾"
    assert all(starts[index + 1] <= ends[index] for index in range(len(parts) - 1)), (
        "相邻块必须重叠或相接,否则漏文本"
    )
    assert all(later > earlier for earlier, later in zip(starts, starts[1:])), (
        "块首必须严格前进,否则重切同一区间(潜在死循环)"
    )
    return starts, ends


def test_overlap_start_never_lands_inside_a_dose_span():
    """overlap 起点落剂量 span 时,下一块必须以完整指令开头,不得出现"次。。"残片。"""
    text = _safe_filler(400) + "甲" * 15 + "一次0.5g一日2次" + _safe_filler(600)
    parts = ClinicalChunker._split_long(text)
    starts, ends = _assert_boundaries_are_clean(text, parts)
    assert len(parts) >= 2, "本用例必须走到多分片路径"
    # 前置条件:天真 overlap 起点(上一块切点 - OVERLAP)确实落在某个 span 中间,
    # 证明本用例真的走到起点保护,而不是什么都没发生。
    naive_start = ends[0] - ClinicalChunker.CHUNK_OVERLAP
    assert any(s < naive_start < e for s, e in _spans(text)), "用例未触发 overlap 起点保护"
    # 保护生效:第二块以完整剂量单位开头,而不是"g一日2次"式残片。
    assert starts[1] < naive_start, "起点必须被推到 span 起点(更早),而非原样保留"
    assert parts[1].startswith("0.5g")


def test_overlap_start_protection_survives_cut_and_start_spans_together():
    """切点保护与起点保护串联:切点先被推过 span,调整后的 overlap 起点再落进另一个 span。"""
    text = (
        _safe_filler(400) + "一次0.5g一日2次" + _safe_filler(11) + "0.5g"
        + _safe_filler(53) + "一日3次" + _safe_filler(700)
    )
    parts = ClinicalChunker._split_long(text)
    starts, ends = _assert_boundaries_are_clean(text, parts)
    # 前置条件一:目标切点被剂量 span 骑跨(切点保护必须先把 end 推出去)。
    assert any(s < ClinicalChunker.CHUNK_MAX_CHARS < e for s, e in _spans(text))
    # 前置条件二:调整后的 overlap 起点落在另一个 span 中间(起点保护必须接手)。
    naive_start = ends[0] - ClinicalChunker.CHUNK_OVERLAP
    assert any(s < naive_start < e for s, e in _spans(text))
    assert parts[1].startswith("0.5g")


def test_split_long_stays_consistent_on_many_interleaved_dose_spans():
    """密布剂量 span 的长文:多次触发两级保护,边界/覆盖/进展不变量仍全部成立。"""
    phrases = ["一次0.5g一日2次", "一次10mg一日3次", "0.25g", "一日2次", "一次2片"]
    segments: list[str] = []
    for index in range(12):
        segments.append(_safe_filler(37 + index * 5))
        segments.append(phrases[index % len(phrases)])
    text = "".join(segments) + _safe_filler(400)
    parts = ClinicalChunker._split_long(text)
    assert len(parts) >= 2, "本用例必须走到多分片路径"
    _assert_boundaries_are_clean(text, parts)


# ------------------------------- 2b: 口语剂量形态的参数化边界回归(真实 pattern)

# 覆盖 W10 扩展的每一种口语形态 + 两个既有形态作对照。全部是对**真实**
# DOSE_PROTECT_PATTERN 声明出来的 span 做边界断言,不复制 pattern。
SPOKEN_DOSE_FORMS = [
    "一次0.5g一日2次",  # 对照:既有阿拉伯数字+单位+频次
    "一次10mg一日3次",  # 对照:西文单位
    "每次半片",  # 半
    "一天两次",  # "一天"同义频次
    "一日两次",  # 中文数字"两"
    "二分之一片",  # 中文分数
    "1/2片",  # ASCII 分数(旧 pattern 只匹配尾部"2片")
    "½片",  # Unicode 分数
    "早晚各一粒",  # 频次副词(无"次")
    "早晚各一次",  # 频次副词 + "次"
]


def _phrase_spans(phrase: str, at: int) -> list[tuple[int, int]]:
    """phrase 放到原文位置 ``at`` 时,真实 pattern 在它上面声明的 span(绝对偏移)。"""
    return [
        (at + match.start(), at + match.end())
        for match in ClinicalChunker.DOSE_PROTECT_PATTERN.finditer(phrase)
    ]


def _placement_covering(phrase: str, target: int) -> int:
    """把 phrase 放到某个起点,使它的某个 span 严格覆盖 ``target``。"""
    for back in range(0, 12):
        at = target - back
        if any(start < target < end for start, end in _phrase_spans(phrase, at)):
            return at
    raise AssertionError(f"{phrase!r} 没有任何 span 能覆盖位置 {target}")


def _spoken_form_text(phrase: str, target: int) -> str:
    """非周期填充 + 覆盖 target 的 phrase + 填充;总长远超两块,target 附近无边界字符。"""
    at = _placement_covering(phrase, target)
    return _safe_filler(at) + phrase + _safe_filler(1200 - at - len(phrase))


@pytest.mark.parametrize("phrase", SPOKEN_DOSE_FORMS)
def test_cut_point_never_splits_a_spoken_dose_form(phrase):
    """各口语剂量形态:硬切点落 span 内时必须被推出 span,不得腰斩给药指令。"""
    target = ClinicalChunker.CHUNK_MAX_CHARS
    text = _spoken_form_text(phrase, target)
    parts = ClinicalChunker._split_long(text)
    _starts, ends = _assert_boundaries_are_clean(text, parts)
    assert len(parts) >= 2, "本用例必须走到多分片路径"
    # 前置条件:目标切点确实被某个 span 骑跨(证明切点保护真的触发过)。
    covering = [(s, e) for s, e in _spans(text) if s < target < e]
    assert covering, f"{phrase!r} 未覆盖切点 {target},用例无效"
    # 切点被推到覆盖 span 的结束处(至少离开 span 内部)。
    assert ends[0] >= covering[0][1]


@pytest.mark.parametrize("phrase", SPOKEN_DOSE_FORMS)
def test_overlap_start_never_splits_a_spoken_dose_form(phrase):
    """各口语剂量形态:overlap 起点落 span 内时,下一块必须以完整指令开头。"""
    target = ClinicalChunker.CHUNK_MAX_CHARS - ClinicalChunker.CHUNK_OVERLAP
    text = _spoken_form_text(phrase, target)
    parts = ClinicalChunker._split_long(text)
    starts, ends = _assert_boundaries_are_clean(text, parts)
    assert len(parts) >= 2, "本用例必须走到多分片路径"
    # 前置条件:天真 overlap 起点(上一块切点 - OVERLAP)确实落在某个 span 中间。
    naive_start = ends[0] - ClinicalChunker.CHUNK_OVERLAP
    assert naive_start == target, "首块切点应未被口语形态干扰(它在 480 处干净落下)"
    covering = [(s, e) for s, e in _spans(text) if s < naive_start < e]
    assert covering, f"{phrase!r} 未覆盖 overlap 起点 {naive_start},用例无效"
    # 起点被推到 span 起点:下一块以完整指令开头,而非"g一日2次"式残片。
    assert starts[1] == covering[0][0]
    assert parts[1].startswith(text[covering[0][0]:covering[0][1]])


# ------------------------------------------------------- 3: reranker 稳定 tie


def test_rerank_ties_keep_input_order_for_value_equal_documents():
    """词法路径:同分文档必须保持输入顺序;值相等的重复文档不得抢首个下标。"""
    reranker = Reranker(enabled=False)
    first = Document("a b x", {"chunk_id": "1"})
    middle = Document("a b y", {"chunk_id": "2"})
    duplicate = Document("a b x", {"chunk_id": "1"})  # 与 first 值相等
    ranked = reranker.rerank("a b", [first, middle, duplicate], top_n=3)
    assert [doc.metadata["chunk_id"] for doc, _ in ranked] == ["1", "2", "1"]


def test_rerank_cross_encoder_path_also_keeps_stable_ties(monkeypatch):
    pytest.importorskip("sentence_transformers")
    import sentence_transformers

    class _EqualScores:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return [0.5 for _ in pairs]

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _EqualScores)
    reranker = Reranker(model_name="wave10-stub")
    docs = [
        Document("same", {"chunk_id": "a"}),
        Document("other", {"chunk_id": "b"}),
        Document("same", {"chunk_id": "a"}),  # 与首个值相等
    ]
    ranked = reranker.rerank("q", docs, top_n=3)
    assert [doc.metadata["chunk_id"] for doc, _ in ranked] == ["a", "b", "a"]
    assert reranker.mode == "cross-encoder"
