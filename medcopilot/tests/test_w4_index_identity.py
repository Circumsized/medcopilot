"""W4.1 / W3.4: manifest v2 嵌入身份契约。

三条硬约束：
1. **不得伪造身份**：通用假 encoder（没有 embedding 身份）不得被写成
   ``med-bge-m3`` / ``unknown``；它只能落到显式的 ``unverified`` 状态。
2. **legacy manifest 永不健康**：v1（无身份字段）可读但永远不算 pinned。
3. **不做"只要数量对就算过"的假通过**：绑定校验必须比较模型/函数名/维度/
   chunker 版本与 live collection 的实际 id 集合，而不仅是 count。
"""

from __future__ import annotations

import json

import pytest

from medcopilot.rag.clinical_chunker import Document
from medcopilot.rag.corpus_store import CorpusStore
from medcopilot.rag.index_manager import IndexManager

LABEL_A = "【药品名称】测试药品A\n【用法用量】口服。一次1片，一日3次。\n【禁忌】过敏者禁用。\n"
LABEL_B = "【药品名称】测试药品B\n【禁忌】孕妇禁用。\n"


def _chunk(label: str, drug: str) -> list[Document]:
    from medcopilot.rag.clinical_chunker import ClinicalChunker

    return ClinicalChunker.chunk_drug_label(label, drug_name=drug)


class GenericFakeEncoder:
    """没有任何嵌入身份的通用假 encoder（测试替身 / 注入工厂的常见形态）。"""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.docs: list[Document] = []

    def index_documents(self, documents) -> int:
        self.docs = list(documents)
        return len(self.docs)

    def count(self) -> int:
        return len(self.docs)

    def __call__(self, text):
        return [0.0, 0.0]


class IdentityEncoder(GenericFakeEncoder):
    """声明了完整嵌入身份（含实测维度）的 encoder。"""

    def __init__(self, collection_name: str, dim: int = 1024, model: str = "BAAI/bge-m3"):
        super().__init__(collection_name)
        self.embedding_dim = dim
        self.embedding_function = _FakeEmbeddingFunction(model)

    def indexed_ids(self) -> list[str]:
        return [d.metadata.get("chunk_id", "") for d in self.docs]


class _FakeEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def name(self) -> str:
        return "med-bge-m3"

    def get_config(self) -> dict:
        return {"model_name": self.model_name}


class CountOnlyLyingEncoder(IdentityEncoder):
    """count 对、但 id 集合与 manifest 记录不一致（count-only 假通过）。"""

    def indexed_ids(self) -> list[str]:
        return ["stale-id-1", "stale-id-2"]


class EmptyLiveEncoder(IdentityEncoder):
    """能枚举但集合已空:indexed_ids=[]、count()=0(内容丢失/从未写入的形态)。"""

    def indexed_ids(self) -> list[str]:
        return []

    def count(self) -> int:
        return 0


def _store(tmp_path) -> CorpusStore:
    store = CorpusStore(path=tmp_path / "corpus.jsonl")
    store.upsert_drug("测试药品A", _chunk(LABEL_A, "测试药品A"))
    return store


def _manager(tmp_path, factory) -> IndexManager:
    return IndexManager(
        corpus_store=_store(tmp_path),
        encoder_factory=factory,
        manifest_path=tmp_path / "index_manifest.json",
        base_collection="med_copilot",
    )


def _boot_encoder(manager: IndexManager, encoder_class=IdentityEncoder, **kwargs):
    """模拟重启后的 boot:一个**装着 manifest 钉住内容**的新 encoder 实例。

    真实 boot 里 collection 是持久化的,新 encoder 打开同一 collection 即持有
    同一批 chunk;fresh empty encoder 什么都不装,那是测试假象而非 boot 形态——
    空集合对着非空 pin 必须判未验证(内容丢失的实证),不是合法 pin。
    """
    encoder = encoder_class(manager.current["collection"], **kwargs)
    encoder.index_documents(manager.active_encoder.docs)
    return encoder


class Factory:
    def __init__(self, encoder_class=IdentityEncoder):
        self.encoder_class = encoder_class
        self.calls: list[str] = []

    def __call__(self, name: str):
        self.calls.append(name)
        return self.encoder_class(name)


# ------------------------------------------------------------------ 身份写入


def test_manifest_records_the_encoders_real_identity(tmp_path):
    manager = _manager(tmp_path, Factory())
    manifest = manager.rebuild()

    assert manifest["embedding_model"] == "BAAI/bge-m3"
    assert manifest["embedding_fn_name"] == "med-bge-m3"
    assert manifest["embedding_dim"] == 1024
    assert manifest["embedding_verified"] is True
    assert manifest["chunker_version"]
    assert manager.manifest_status == "ok"


def test_manifest_never_invents_an_identity_for_generic_encoders(tmp_path):
    """通用假 encoder 的身份字段必须是空 + 显式 unverified，不是编造的模型名。"""
    manager = _manager(tmp_path, Factory(GenericFakeEncoder))
    manifest = manager.rebuild()

    assert manifest["embedding_model"] is None
    assert manifest["embedding_fn_name"] is None
    assert manifest["embedding_dim"] is None
    assert manifest["embedding_verified"] is False
    assert manager.manifest_status == "unverified"
    assert manager.manifest_error
    # 绝不出现编造的身份字符串。
    for key in ("embedding_model", "embedding_fn_name"):
        assert manifest[key] not in ("med-bge-m3", "unknown", "bge-m3")


def test_manifest_records_the_index_content_digest_when_available(tmp_path):
    manager = _manager(tmp_path, Factory())
    manifest = manager.rebuild()
    assert manifest["index_content_sha256"]
    # 同一内容两次重建 → 指纹稳定。
    assert manager.rebuild()["index_content_sha256"] == manifest["index_content_sha256"]


# ------------------------------------------------------------------ 加载校验


def test_legacy_manifest_is_never_a_healthy_pin(tmp_path):
    manifest_path = tmp_path / "index_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "collection": "med_copilot-blue",
                "corpus_sha256": "a" * 64,
                "corpus_chunk_count": 3,
            }
        ),
        encoding="utf-8",
    )
    manager = _manager(tmp_path, Factory())
    assert manager.current is not None, "legacy manifest 仍可读（迁移用）"
    assert manager.manifest_status == "legacy"
    assert "missing embedding identity" in (manager.manifest_error or "")

    manager.bind_active(IdentityEncoder("med_copilot-blue"), "med_copilot-blue")
    assert manager.pin_verified is False


def test_unverified_manifest_never_binds_as_verified(tmp_path):
    manager = _manager(tmp_path, Factory(GenericFakeEncoder))
    manager.rebuild()
    assert manager.manifest_status == "unverified"
    manager.bind_active(GenericFakeEncoder(manager.current["collection"]),
                        manager.current["collection"])
    assert manager.pin_verified is False


def test_malformed_identity_field_is_invalid_not_silently_defaulted(tmp_path):
    manifest_path = tmp_path / "index_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "collection": "med_copilot-blue",
                "corpus_sha256": "a" * 64,
                "corpus_chunk_count": 3,
                "embedding_model": "BAAI/bge-m3",
                "embedding_fn_name": "med-bge-m3",
                "embedding_dim": "big",
                "chunker_version": "chunker-v2",
            }
        ),
        encoding="utf-8",
    )
    manager = _manager(tmp_path, Factory())
    assert manager.manifest_status == "invalid"
    assert "embedding_dim" in (manager.manifest_error or "")


def test_corrupt_manifest_stays_invalid(tmp_path):
    (tmp_path / "index_manifest.json").write_text("{not json", encoding="utf-8")
    manager = _manager(tmp_path, Factory())
    assert manager.manifest_status == "invalid"
    assert manager.manifest_error


# ------------------------------------------------------------------ 绑定校验


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda e: setattr(e, "embedding_dim", 768), "embedding_dim"),
        (lambda e: setattr(e.embedding_function, "model_name", "other/model"), "embedding_model"),
    ],
)
def test_bind_active_rejects_identity_mismatch(tmp_path, mutate, field):
    manager = _manager(tmp_path, Factory())
    manifest = manager.rebuild()
    encoder = IdentityEncoder(manifest["collection"])
    mutate(encoder)
    manager.bind_active(encoder, manifest["collection"])
    assert manager.pin_verified is False


def test_bind_active_rejects_chunker_version_mismatch(tmp_path):
    manager = _manager(tmp_path, Factory())
    manifest = manager.rebuild()
    tampered = dict(manifest)
    tampered["chunker_version"] = "chunker-v1"
    manager._manifest = tampered  # 手工漂移：manifest 声明了别的切分器版本
    manager.bind_active(IdentityEncoder(manifest["collection"]), manifest["collection"])
    assert manager.pin_verified is False


def test_bind_active_accepts_a_matching_pin(tmp_path):
    manager = _manager(tmp_path, Factory())
    manifest = manager.rebuild()
    manager.bind_active(_boot_encoder(manager), manifest["collection"])
    assert manager.pin_verified is True


def test_bind_active_is_not_fooled_by_a_count_only_match(tmp_path):
    """count 相等但 live id 集合与 manifest 指纹不符 → 不得判为 verified。"""
    manager = _manager(tmp_path, Factory())
    manifest = manager.rebuild()
    manager.bind_active(
        CountOnlyLyingEncoder(manifest["collection"]), manifest["collection"]
    )
    assert manager.pin_verified is False


def test_bind_active_ignores_content_check_when_digest_is_absent(tmp_path):
    """无 index_content_sha256 的后端（老 encoder 协议）仍走维度/身份/数量校验。"""
    manager = _manager(tmp_path, Factory())
    manifest = manager.rebuild()
    manifest.pop("index_content_sha256")
    manager._manifest = manifest
    manager.bind_active(_boot_encoder(manager), manifest["collection"])
    assert manager.pin_verified is True


def test_bind_active_rejects_an_empty_live_collection(tmp_path):
    """能枚举但集合为空(indexed_ids=[]、count=0)而 manifest 钉着 chunk ⇒ 未验证。

    ``if live_ids:`` 跳过空列表、``elif live_ids is None`` 也不接空列表,旧代码
    让空集合直接 fall through 到 True——空集合对着非空 pin 是"内容丢了"的实证,
    不是"后端尚未加载"的中性状态,不得警告/静默之后仍报 verified。
    """
    manager = _manager(tmp_path, Factory())
    manifest = manager.rebuild()
    manager.bind_active(_boot_encoder(manager, EmptyLiveEncoder), manifest["collection"])
    assert manager.pin_verified is False


def test_bind_active_rejects_an_empty_live_collection_without_a_digest(tmp_path):
    """无 digest 的老协议下,空枚举走 count 校验:0 != 钉住数量 ⇒ 未验证。"""
    manager = _manager(tmp_path, Factory())
    manifest = manager.rebuild()
    manifest.pop("index_content_sha256")
    manager._manifest = manifest
    manager.bind_active(_boot_encoder(manager, EmptyLiveEncoder), manifest["collection"])
    assert manager.pin_verified is False


# ------------------------------------------------------------------ 重建发布点


def test_rebuild_refreshes_pin_verified_for_a_real_identity(tmp_path):
    """fresh 无 manifest → 成功重建后 pin_verified 必须刷新。

    此前 rebuild 只写 manifest_status,pin_verified 保持 None:manifest ok 但
    /health 因 ``pin_verified is not True`` 永远 not-ready,直到进程重启。
    """
    manager = _manager(tmp_path, Factory())
    assert manager.pin_verified is None
    manager.rebuild()
    assert manager.manifest_status == "ok"
    assert manager.pin_verified is True


def test_rebuild_keeps_pin_verified_false_for_an_unverified_identity(tmp_path):
    """stub(无嵌入身份)重建成功:manifest 如实 unverified,pin 不得假冒 True。"""
    manager = _manager(tmp_path, Factory(GenericFakeEncoder))
    manager.rebuild()
    assert manager.manifest_status == "unverified"
    assert manager.pin_verified is False


def test_failed_rebuild_keeps_the_previous_pin_state(tmp_path):
    """重建失败:pin_verified 原样保持(从未成功过是 None;成功过之后是 True)。"""

    class FlakyEncoder(IdentityEncoder):
        def index_documents(self, documents):
            raise RuntimeError("simulated backend failure")

    class FlakyFactory:
        def __call__(self, name: str):
            return FlakyEncoder(name)

    manager = _manager(tmp_path, FlakyFactory())
    with pytest.raises(RuntimeError):
        manager.rebuild()
    assert manager.pin_verified is None, "从未成功过:保持初始 None"

    class FlakyAfterFirst:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, name: str):
            self.calls += 1
            if self.calls == 1:
                return IdentityEncoder(name)
            raise RuntimeError("simulated backend failure")

    manager2 = _manager(tmp_path, FlakyAfterFirst())
    manager2.rebuild()
    assert manager2.pin_verified is True
    with pytest.raises(RuntimeError):
        manager2.rebuild()
    assert manager2.pin_verified is True, "失败不得把已发布的 pin 拖假"


def test_content_digest_mismatch_fails_the_rebuild(tmp_path):
    """index_documents 之后的 live id 集合必须等于预期集合，否则重建失败。"""

    class HalfWrittenEncoder(IdentityEncoder):
        def indexed_ids(self):
            return [d.metadata.get("chunk_id", "") for d in self.docs[:-1]]

    manager = _manager(tmp_path, Factory(HalfWrittenEncoder))
    with pytest.raises(Exception):
        manager.rebuild()
    assert manager.current is None, "重建失败必须保持原状（无 manifest 可回滚）"


def test_rebuild_rollback_keeps_the_previous_pin(tmp_path):
    class FlakyEncoder(IdentityEncoder):
        def __init__(self, name, registry=None):
            super().__init__(name)
            self.registry = registry

        def index_documents(self, documents):
            if self.registry and len(self.registry.encoders) > 1:
                raise RuntimeError("simulated backend failure")
            return super().index_documents(documents)

    class TrackingFactory:
        def __init__(self):
            self.encoders: list[GenericFakeEncoder] = []

        def __call__(self, name: str):
            encoder = FlakyEncoder(name, registry=self)
            self.encoders.append(encoder)
            return encoder

    factory = TrackingFactory()
    manager = _manager(tmp_path, factory)
    first = manager.rebuild()
    with pytest.raises(RuntimeError):
        manager.rebuild()
    assert manager.current == first
    assert manager.active_encoder.collection_name == first["collection"]
