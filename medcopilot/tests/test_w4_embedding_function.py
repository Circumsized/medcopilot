"""W4.1: MedBgeM3EmbeddingFunction 的 Chroma 1.x 官方契约 + 离线优先语义。

全部离线：模型加载被注入伪造的 SentenceTransformer / 伪造 encode 结果，
不下载任何权重。重点是三条真实缺陷：

1. ``name`` 原先是**字符串**，chromadb 1.4.1 在
   ``validate_embedding_function_conflict_on_get`` 里调用
   ``embedding_function.name()`` → ``TypeError: 'str' object is not callable``，
   ``build_encoder`` 随即返回 None（dense 永远起不来）。
2. 该契约必须**不继承** chromadb 的 ``EmbeddingFunction``：其
   ``__init_subclass__`` 会把 ``__call__`` 包一层
   ``normalize_embeddings``（numpy 化、非空断言），破坏本类对 ``str`` /
   裸 list 入参的兼容性。
3. ``embedding_dim`` 必须是**实测**值，不能拿 ``config.embedding_dim``
   （期望值）冒充实测。
"""

from __future__ import annotations

import json

import pytest

from medcopilot.common.config import config
from medcopilot.rag.vector_store import (
    MED_BGE_M3_FN_NAME,
    ChromaDenseEncoder,
    MedBgeM3EmbeddingFunction,
)

# The dense-encoder weight paths are exercised only where the optional heavy
# stack (sentence-transformers/torch) is installed; CI installs the rag lock
# on the 3.12 job only, so older interpreters skip the whole module.
pytest.importorskip("sentence_transformers")

DIM = 4


class FakeModel:
    """确定性离线替身：encode 返回固定 4 维向量。"""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [[1.0, 2.0, 3.0, 4.0][: self.dim] for _ in texts]


def _stub_load(monkeypatch, model=None) -> FakeModel:
    fake = model or FakeModel()
    monkeypatch.setattr(MedBgeM3EmbeddingFunction, "_load", lambda self: fake)
    return fake


def _patch_sentence_transformer(monkeypatch):
    """把 sentence_transformers.SentenceTransformer 换成记录 kwargs 的替身。"""
    import sentence_transformers

    calls: list[dict] = []

    class _FakeST:
        def __init__(self, model_name, **kwargs):
            calls.append({"kind": "load", "model_name": model_name, **kwargs})
            self.model_name = model_name

        def encode(self, texts, **kwargs):
            calls.append({"kind": "encode", "texts": list(texts), **kwargs})
            return [[1.0, 2.0, 3.0, 4.0] for _ in texts]

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeST)
    return calls


# ------------------------------------------------------- chroma 官方持久化契约


def test_name_is_a_callable_returning_the_registry_name():
    """chromadb 1.4.1 调用 ``name()``（方法），不是字符串属性。"""
    fn = MedBgeM3EmbeddingFunction(model_name="offline-fake")
    assert callable(MedBgeM3EmbeddingFunction.name)
    assert fn.name() == MED_BGE_M3_FN_NAME == "med-bge-m3"
    assert not isinstance(MedBgeM3EmbeddingFunction.name, str)


def test_name_is_a_classmethod_so_chroma_can_register_the_class():
    """chroma 的 ``register_embedding_function(type(ef))`` 在**类**上调 name()。"""
    assert MedBgeM3EmbeddingFunction.name() == MED_BGE_M3_FN_NAME
    assert callable(MedBgeM3EmbeddingFunction.build_from_config)


def test_persistence_contract_round_trips_and_is_json_safe():
    fn = MedBgeM3EmbeddingFunction(model_name="BAAI/bge-m3", batch_size=8, normalize=True)
    assert fn.is_legacy() is False
    assert fn.default_space() == "cosine"
    for space in ("cosine", "l2", "ip"):
        assert space in fn.supported_spaces()
    cfg = fn.get_config()
    # 该 config 会被 chroma 原样写进 sqlite：必须 JSON 可序列化。
    assert json.loads(json.dumps(cfg)) == cfg
    rebuilt = MedBgeM3EmbeddingFunction.build_from_config(cfg)
    assert rebuilt.get_config() == cfg
    assert rebuilt.model_name == "BAAI/bge-m3"


def test_does_not_subclass_chroma_embedding_function():
    """不得继承 chroma 的 EmbeddingFunction：其 __init_subclass__ 会包装 __call__。

    包装后的 __call__ 走 normalize_embeddings（numpy 化 + 非空断言），会破坏
    本类对 str / 裸 list 的兼容性，也让每条嵌入多付一次 numpy 往返。
    """
    bases = [m.__module__.split(".")[0] for m in MedBgeM3EmbeddingFunction.__mro__]
    assert "chromadb" not in bases
    # chroma 的包装器定义在 EmbeddingFunction.__init_subclass__.<locals> 内。
    assert MedBgeM3EmbeddingFunction.__call__.__qualname__.startswith(
        "MedBgeM3EmbeddingFunction."
    )


def test_registration_is_registered_when_chromadb_is_present():
    pytest.importorskip("chromadb")
    from chromadb.utils.embedding_functions import known_embedding_functions

    assert known_embedding_functions.get(MED_BGE_M3_FN_NAME) is MedBgeM3EmbeddingFunction


def test_registration_is_optional_and_never_crashes_without_chromadb(monkeypatch):
    """chromadb 缺失时注册是 no-op，类本身必须仍然可用。"""
    import sys

    from medcopilot.rag import vector_store

    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", None)
    assert vector_store._register_with_chroma(MedBgeM3EmbeddingFunction) is False
    # chroma 不存在也不影响 embedding 语义本身。
    fn = MedBgeM3EmbeddingFunction(model_name="offline-fake")
    _stub_load(monkeypatch)
    assert fn("药品") == [1.0, 2.0, 3.0, 4.0]


# ------------------------------------------------------------------ 向量与维度


def test_call_returns_plain_python_lists_for_str_and_batch(monkeypatch):
    _stub_load(monkeypatch)
    fn = MedBgeM3EmbeddingFunction(model_name="offline-fake", batch_size=2)
    single = fn("药品")
    batch = fn(["药品", "禁忌"])
    assert single == [1.0, 2.0, 3.0, 4.0]
    assert batch == [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    for vector in [single, *batch]:
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)


def test_embed_query_always_returns_a_list_of_vectors(monkeypatch):
    """chroma 的 query 路径调用 ``embed_query``，契约要求 list[embedding]。"""
    _stub_load(monkeypatch)
    fn = MedBgeM3EmbeddingFunction(model_name="offline-fake")
    assert fn.embed_query(["药品"]) == [[1.0, 2.0, 3.0, 4.0]]
    assert fn.embed_query("药品") == [[1.0, 2.0, 3.0, 4.0]]


@pytest.mark.parametrize(
    ("texts", "vectors", "message"),
    [
        (["药品"], [[1.0, "a", 3.0, 4.0]], "non-numeric"),
        (["药品"], [[]], "empty"),
        (["药品"], [[1.0, float("nan"), 3.0, 4.0]], "finite"),
        (["药品"], [[1.0, float("inf"), 3.0, 4.0]], "finite"),
        (["药品", "禁忌"], [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], "inconsistent"),
    ],
)
def test_vectors_are_validated_numeric_nonempty_finite_and_equal_dim(
    monkeypatch, texts, vectors, message
):
    """坏向量必须显式报错，而不是悄悄写进索引变成假混合。"""

    class BadModel:
        def encode(self, _texts, **_kwargs):
            return [list(v) for v in vectors]

    fn = MedBgeM3EmbeddingFunction(model_name="offline-fake")
    _stub_load(monkeypatch, BadModel())
    with pytest.raises((ValueError, TypeError)) as exc:
        fn(texts)
    assert message in str(exc.value).lower()
    assert fn.embedding_dim is None, "坏向量不得缓存维度"


def test_wrong_vector_count_is_rejected(monkeypatch):
    class ShortModel:
        def encode(self, texts, **kwargs):
            return [[1.0, 2.0, 3.0, 4.0]]

    fn = MedBgeM3EmbeddingFunction(model_name="offline-fake")
    _stub_load(monkeypatch, ShortModel())
    with pytest.raises(ValueError, match="count"):
        fn(["a", "b"])


def test_embedding_dim_is_measured_not_configured(monkeypatch):
    """config.embedding_dim 是期望值；实测维度只能来自一次真实 encode。"""
    fn = MedBgeM3EmbeddingFunction(model_name="offline-fake")
    assert fn.embedding_dim is None
    assert fn.dim_source == "unmeasured"
    assert fn.expected_dim == config.embedding_dim == 1024
    assert fn.expected_dim != fn.embedding_dim

    _stub_load(monkeypatch)
    fn(["药品"])
    assert fn.embedding_dim == DIM
    assert fn.dim_source == "measured"


def test_measure_dim_probes_once_and_caches(monkeypatch):
    fake = _stub_load(monkeypatch)
    fn = MedBgeM3EmbeddingFunction(model_name="offline-fake")
    assert fn.measure_dim() == DIM
    assert fn.measure_dim() == DIM
    assert len(fake.calls) == 1, "维度探测必须只跑一次"
    assert fn.embedding_dim == DIM


def test_measure_dim_returns_none_when_the_model_is_unavailable(monkeypatch):
    """离线且无本地快照 ⇒ 维度探测失败，状态如实报 load_failed。"""
    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers,
        "SentenceTransformer",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no local snapshot")),
    )
    fn = MedBgeM3EmbeddingFunction(model_name="missing/offline-model")
    assert fn.measure_dim() is None
    assert fn.dim_source == "load_failed"
    assert fn.embedding_dim is None
    assert fn.failed is True


def test_unavailable_model_raises_explicitly(monkeypatch):
    monkeypatch.setattr(MedBgeM3EmbeddingFunction, "_load", lambda self: None)
    fn = MedBgeM3EmbeddingFunction(model_name="missing/offline-model")
    with pytest.raises(RuntimeError, match="embedding model unavailable"):
        fn(["药品"])


# -------------------------------------------------------------------- 离线优先


def test_load_is_offline_first_and_uses_the_configured_cache(monkeypatch):
    """默认禁止运行时下载：local_files_only 来自 config.model_weights_offline。"""
    monkeypatch.setattr(config, "model_weights_offline", True)
    monkeypatch.setattr(config, "embedding_cache_dir", "/opt/hf-cache")
    calls = _patch_sentence_transformer(monkeypatch)

    fn = MedBgeM3EmbeddingFunction(model_name="BAAI/bge-m3", batch_size=7, normalize=False)
    assert fn.local_files_only is True
    assert fn.cache_dir == "/opt/hf-cache"
    fn(["药品"])

    load = calls[0]
    encode = calls[1]
    assert load["kind"] == "load" and load["model_name"] == "BAAI/bge-m3"
    assert load["local_files_only"] is True
    assert load["cache_folder"] == "/opt/hf-cache"
    assert encode["batch_size"] == 7
    assert encode["normalize_embeddings"] is False
    assert encode["texts"] == ["药品"]


def test_offline_mode_can_be_relaxed_explicitly(monkeypatch):
    monkeypatch.setattr(config, "model_weights_offline", False)
    calls = _patch_sentence_transformer(monkeypatch)
    fn = MedBgeM3EmbeddingFunction(model_name="BAAI/bge-m3")
    fn(["药品"])
    assert calls[0]["local_files_only"] is False


def test_defaults_come_from_app_config(monkeypatch):
    monkeypatch.setattr(config, "embedding_model", "BAAI/bge-m3")
    monkeypatch.setattr(config, "embedding_batch_size", 32)
    monkeypatch.setattr(config, "embedding_normalize", False)
    fn = MedBgeM3EmbeddingFunction()
    assert fn.model_name == "BAAI/bge-m3"
    assert fn.batch_size == 32
    assert fn.normalize is False


def test_status_reports_health_without_loading_a_model():
    fn = MedBgeM3EmbeddingFunction(model_name="BAAI/bge-m3")
    status = fn.status()
    assert status["available"] is False
    assert status["model"] == "BAAI/bge-m3"
    assert status["embedding_dim"] is None
    assert status["dim_source"] == "unmeasured"
    assert status["expected_dim"] == config.embedding_dim
    assert status["local_files_only"] is True
    assert fn._model is None, "status 查询不得触发模型加载"


# --------------------------------------------------- 离线 stub 的持久化 pin 测试

LABEL_A = (
    "【药品名称】测试药品A\n【禁忌】活动期消化道溃疡及出血患者禁用。\n"
    "【用法用量】口服。一次1片，一日3次。\n"
    "【药物相互作用】丙磺舒可提高本品血药浓度。\n"
)


def _chunks():
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    return ClinicalChunker.chunk_drug_label(LABEL_A, drug_name="测试药品A")


def test_chroma_collection_create_and_reopen_pins_persistence(tmp_path, monkeypatch):
    """真实 chromadb + 离线 stub encoder 的落盘/重开契约（不下载权重）。"""
    chromadb = pytest.importorskip("chromadb")
    from medcopilot.rag.vector_store import ChromaDenseEncoder

    chunks = _chunks()
    # 类级打桩：build_from_config 重建出的新实例也能拿到同一个假模型。
    monkeypatch.setattr(MedBgeM3EmbeddingFunction, "_load", lambda self: FakeModel())
    path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(path))
    fn = MedBgeM3EmbeddingFunction(model_name="offline-fake")
    collection = client.get_or_create_collection(
        name="med_copilot-blue", embedding_function=fn
    )
    encoder = ChromaDenseEncoder(
        collection, fn, collection_name="med_copilot-blue", client=client
    )
    assert encoder.index_documents(chunks) == len(chunks)

    # 1) 集合身份必须被 chroma 以官方格式持久化（name + config），space 来自
    #    default_space()。
    reopened_config = client.get_collection(
        name="med_copilot-blue", embedding_function=None
    ).configuration_json
    assert reopened_config["embedding_function"]["type"] == "known"
    assert reopened_config["embedding_function"]["name"] == MED_BGE_M3_FN_NAME
    assert reopened_config["embedding_function"]["config"]["model_name"] == "offline-fake"
    assert reopened_config["hnsw"]["space"] == "cosine"

    # 2) 全新 client（同一进程、同一路径）重开：chunk 数与 id 集合必须原样恢复。
    reopened_client = chromadb.PersistentClient(path=str(path))
    reopened = reopened_client.get_collection(
        name="med_copilot-blue", embedding_function=None
    )
    assert reopened.count() == len(chunks)
    assert sorted(reopened.get(include=[])["ids"]) == sorted(
        doc.metadata["chunk_id"] for doc in chunks
    )
    # 3) 查询路径同样可用：EF 是从持久化 config 经 build_from_config 重建的。
    assert reopened.query(query_texts=["禁忌"], n_results=1)["ids"]


def test_index_documents_rejects_a_backend_that_drops_ids(monkeypatch):
    """count 相等但 id 集合不对 = 假通过；必须显式失败（无 count-only false pass）。"""
    pytest.importorskip("chromadb")
    from medcopilot.rag.vector_store import ChromaDenseEncoder

    chunks = _chunks()

    class DroppingCollection:
        def count(self):
            return len(chunks)

        def get(self, include=None):
            return {"ids": ["only-one"]}

        def upsert(self, **kwargs):
            return None

    class DroppingClient:
        def get_or_create_collection(self, **kwargs):
            return DroppingCollection()

        def delete_collection(self, name):
            return None

    _stub_load(monkeypatch)
    encoder = ChromaDenseEncoder(
        DroppingCollection(),
        MedBgeM3EmbeddingFunction(model_name="offline-fake"),
        collection_name="med_copilot-blue",
        client=DroppingClient(),
    )
    with pytest.raises(RuntimeError, match="index validation failed"):
        encoder.index_documents(chunks)


def test_build_encoder_returns_none_when_chromadb_is_absent(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "chromadb", None)
    assert (
        ChromaDenseEncoder.build_encoder(
            persist_directory=str(tmp_path), collection_name="c"
        )
        is None
    )
