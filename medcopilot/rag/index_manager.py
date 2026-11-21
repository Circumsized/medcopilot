"""稠密索引的 Manifest 管理与蓝绿重建（Slice 3）。

权威语料（CorpusStore）是唯一事实来源；稠密索引（Chroma collection）是
可删除、可重建的派生物。重建走蓝绿流程：向一个全新的 collection 全量灌入
权威语料 → 校验数量一致 → 原子写 manifest 切换活跃指针。任何构建或校验
失败都保持旧 manifest 与活跃 encoder 不变，旧 collection 天然承担回滚角色。

IndexManager 不绑定具体向量库：encoder 协议只需
``index_documents(docs) -> int``、``count() -> int``、``collection_name``。
``index_documents`` 必须实现为**全量替换**（数量校验依赖这一语义），并且仅在
数据已持久化后才返回——manifest 在其后才落盘，"先数据后指针"的持久化顺序由
encoder 后端保证（Chroma 的 upsert 提交 sqlite 队列后才返回，满足该前提）。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Callable

from medcopilot.common.fsutil import durable_replace
from medcopilot.common.logger import logger
from medcopilot.rag.clinical_chunker import ClinicalChunker
from medcopilot.rag.corpus_store import CorpusStore
from medcopilot.rag.encoder_protocol import DenseIndexEncoder

BLUE_SUFFIX = "-blue"
GREEN_SUFFIX = "-green"


def _resolve_identity(source, attribute: str):
    """Read an identity attribute, calling it when it is a method/property.

    ``MedBgeM3EmbeddingFunction.name`` is a classmethod (chroma's registry calls
    it as one), so a plain ``getattr`` yields a bound method. Comparing that to
    the manifest's string would mark every healthy boot as an unverified pin.
    """
    if source is None:
        return None
    value = getattr(source, attribute, None)
    if callable(value):
        try:
            value = value()
        except Exception as exc:
            logger.warning("Identity probe {}.{} failed ({}).", type(source).__name__, attribute, exc)
            return None
    return value if isinstance(value, (str, int, float, type(None))) else None


# Keys a valid manifest must carry; anything else is a corrupted (not merely
# absent) manifest and must be reported as such instead of silently pinning
# to partial data via .get() defaults.
MANIFEST_REQUIRED_KEYS = ("schema_version", "collection", "corpus_sha256", "corpus_chunk_count")
# W4 embedding identity: v2 manifests pin the model/function/dimension so a
# rebuilt collection cannot silently be queried with a different embedding space.
MANIFEST_EMBEDDING_KEYS = ("embedding_model", "embedding_fn_name", "embedding_dim", "chunker_version")


class IndexValidationError(Exception):
    """重建后的索引与权威语料不一致；活跃索引保持不变。"""


def _encoder_embedding_function(encoder):
    """The encoder's embedding function, via its public API first."""
    function = getattr(encoder, "embedding_function", None)
    if function is None:
        function = getattr(encoder, "_embedding_fn", None)
    return function


def _snapshot_ids(snapshot) -> list[str]:
    return [
        str(doc.metadata.get("chunk_id") or f"doc-{index}")
        for index, doc in enumerate(snapshot)
    ]


def _ids_digest(ids) -> str:
    digest = hashlib.sha256()
    for chunk_id in sorted(str(item) for item in ids):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _encoder_content_digest(encoder) -> str | None:
    """Content fingerprint of what the backend actually holds (None if unknown).

    ``count()`` alone cannot distinguish "the right chunks" from "the right
    number of wrong chunks", so backends that can enumerate ids are asked to.
    """
    ids = _encoder_live_ids(encoder)
    return None if ids is None else _ids_digest(ids)


def _encoder_live_ids(encoder) -> list[str] | None:
    """Live chunk ids, or None when the backend cannot enumerate them.

    An empty list is deliberately NOT the same as None but is also not treated as
    proof of an empty collection: a backend that has not loaded yet (and a fresh
    test double) both enumerate nothing while the manifest pins real content. Only
    a non-empty enumeration is used as evidence — see ``bind_active``.
    """
    indexed_ids = getattr(encoder, "indexed_ids", None)
    if not callable(indexed_ids):
        return None
    try:
        return [str(item) for item in indexed_ids()]
    except Exception as exc:
        logger.warning("Encoder id enumeration unavailable ({}); falling back to count.", exc)
        return None


def _encoder_identity(encoder) -> dict:
    """Embedding identity of the backend, reported field by field.

    A generic test double (or an injected factory) has no embedding identity at
    all. Inventing one — "unknown", or worse the default model name — would
    record a claim nobody can check, so unresolvable fields stay None and
    ``embedding_verified`` says so.

    The fields are deliberately **independent of ``embedding_verified``**: a
    production encoder at boot exposes its model/function names before any
    encode has measured the dimension (``embedding_dim`` stays None until
    then). Gating the names behind the dimension used to blank the whole
    identity at boot, so a swapped model compared equal to the manifest's pin
    and the check reported a healthy green — exactly the false pass this
    verification exists to catch.
    """
    measure = getattr(encoder, "measure_dim", None)
    if callable(measure):
        # Best effort: a real encoder reports its *measured* dimension here, which
        # is what makes the record checkable at all.
        try:
            measure()
        except Exception as exc:
            logger.warning("Dimension probe before manifest write failed ({}).", exc)
    function = _encoder_embedding_function(encoder)
    model = _resolve_identity(function, "model_name")
    fn_name = _resolve_identity(function, "name")
    api_base = _resolve_identity(function, "api_base")
    dim = _resolve_identity(encoder, "embedding_dim")
    dim = int(dim) if isinstance(dim, (int, float)) and not isinstance(dim, bool) and dim > 0 else None
    verified = model is not None and fn_name is not None and dim is not None
    return {
        "embedding_model": str(model) if model is not None else None,
        "embedding_fn_name": str(fn_name) if fn_name is not None else None,
        "embedding_endpoint": str(api_base) if api_base else None,
        "embedding_dim": dim,
        "embedding_verified": bool(verified),
    }

class IndexManager:
    def __init__(
        self,
        *,
        corpus_store: CorpusStore,
        encoder_factory: Callable[[str], DenseIndexEncoder | None],
        manifest_path: str | Path,
        base_collection: str = "med_copilot",
    ) -> None:
        self._corpus_store = corpus_store
        self._encoder_factory = encoder_factory
        self._manifest_path = Path(manifest_path)
        self._base_collection = base_collection
        self._lock = threading.Lock()
        # Boot health surface (read by /health and /index/status); defaults
        # here are refined by _load_manifest() below:
        # manifest_status: "ok" | "absent" | "invalid" — a corrupted manifest
        # is a distinct state from no manifest; treating it as absent left
        # /health all-green while the blue-green pin relationship was lost.
        # pin_verified: True only when the bound collection matched the
        # manifest and its live chunk count equaled the pinned count.
        self.manifest_status: str = "absent"
        self.manifest_error: str | None = None
        self.pin_verified: bool | None = None
        self._manifest: dict | None = self._load_manifest()
        self._active_encoder = None

    # ------------------------------------------------------------------ state

    @property
    def current(self) -> dict | None:
        return self._manifest

    @property
    def active_encoder(self) -> DenseIndexEncoder | None:
        return self._active_encoder

    def bind_active(self, encoder, collection_name: str) -> None:
        """进程启动时把既有活跃 encoder 绑定到管理器（不写 manifest）。"""
        with self._lock:
            self._active_encoder = encoder
            if self._manifest is None:
                self.pin_verified = False if self.manifest_status in {"invalid", "legacy", "unverified"} else None
                return
            if self.manifest_status in {"invalid", "legacy", "unverified"}:
                # Readable, but it proves nothing about identity or content, so it
                # must never present itself as a healthy pin.
                self.pin_verified = False
                return
            if collection_name != self._manifest.get("collection"):
                logger.warning(
                    "Active encoder collection '{}' does not match manifest '{}'.",
                    collection_name, self._manifest.get("collection"),
                )
                self.pin_verified = False
                return
            identity = _encoder_identity(encoder)
            if (
                identity["embedding_model"] is None
                and identity["embedding_fn_name"] is None
                and identity["embedding_dim"] is None
            ):
                # The backend exposes no embedding identity at all: nothing
                # observed here can prove that query vectors and stored vectors
                # share a space, so the pin must not read as verified — a
                # matching count alone is not identity.
                logger.warning(
                    "Active encoder for '{}' exposes no embedding identity; "
                    "treating the pin as unverified.",
                    collection_name,
                )
                self.pin_verified = False
                return
            expected = {
                "embedding_model": identity["embedding_model"],
                "embedding_fn_name": identity["embedding_fn_name"],
                "embedding_endpoint": identity["embedding_endpoint"],
                "embedding_dim": identity["embedding_dim"],
                # A changed chunker invalidates every stored chunk boundary, so a
                # collection built by an older chunker must not read as pinned.
                "chunker_version": getattr(ClinicalChunker, "VERSION", None),
            }
            # Field-by-field: each live identity value is compared on its own.
            # An unmeasurable dimension (None until the first encode) must NOT
            # take the model/function comparison down with it — a swapped model
            # with an unmeasured dim is precisely the boot-time false green.
            for key, actual in expected.items():
                if actual is not None and self._manifest.get(key) not in (None, actual):
                    logger.warning(
                        "Active encoder {} '{}' does not match manifest pin '{}'.",
                        key, actual, self._manifest.get(key),
                    )
                    self.pin_verified = False
                    return
            live_ids = _encoder_live_ids(encoder)
            pinned_digest = self._manifest.get("index_content_sha256")
            if pinned_digest and live_ids is not None:
                # The backend CAN enumerate its ids (possibly empty) and the
                # manifest pins a content digest: the pinned set is the only
                # acceptable one. An empty enumeration against a manifest that
                # pins chunks mismatches exactly like a stale id set — ``if
                # live_ids:`` used to skip the empty case and the ``elif ...
                # is None`` branch did not catch it either, so an empty
                # collection fell straight through to pin_verified=True.
                if _ids_digest(live_ids) != pinned_digest:
                    logger.warning(
                        "Collection '{}' holds a different chunk id set than the manifest pins "
                        "(count-only match); treating the pin as unverified.",
                        collection_name,
                    )
                    self.pin_verified = False
                    return
            elif live_ids is None or not live_ids:
                # No digest to compare against (or nothing enumerated to compare
                # with): the chunk count is the only content evidence left. This
                # covers the structural-encoder protocol (None) AND the empty
                # enumeration — an empty collection against a manifest that pins
                # chunks must not read as verified.
                try:
                    live = encoder.count()
                except Exception as exc:
                    # A count probe that itself fails (corrupt store, lock
                    # contention) must not silently skip the health check — the
                    # pin would present as verified when it never was.
                    logger.warning(
                        "Pin health check for '{}' failed ({}); treating as unverified.",
                        collection_name, exc,
                    )
                    self.pin_verified = False
                    return
                if live != self._manifest.get("corpus_chunk_count"):
                    logger.warning(
                        "Collection '{}' holds {} chunks but the manifest pins {}; "
                        "dense index is stale — run POST /index/rebuild.",
                        collection_name, live, self._manifest.get("corpus_chunk_count"),
                    )
                    self.pin_verified = False
                    return
            # else: a non-empty enumeration with no digest recorded (older
            # encoder protocol): identity + collection + chunker checks above are
            # the only available evidence. Corpus drift is reported separately by
            # the boot-time corpus-hash check and /index/status.
            self.pin_verified = True

    # ---------------------------------------------------------------- rebuild

    def rebuild(self) -> dict:
        """从权威语料全量重建索引并蓝绿切换；失败时一切保持原状。"""
        with self._lock:
            snapshot = self._corpus_store.snapshot
            corpus_hash = self._corpus_hash(snapshot)
            next_name = self._next_collection_name()
            encoder = self._encoder_factory(next_name)
            if encoder is None:
                raise ValueError("encoder factory returned no backend for " + next_name)
            try:
                encoder.index_documents(snapshot)
                indexed = encoder.count()
            except Exception as exc:
                logger.error("Index rebuild into '{}' failed ({}); keeping active index.", next_name, exc)
                raise
            if indexed != len(snapshot):
                raise IndexValidationError(
                    f"index validation failed: {indexed} indexed vs {len(snapshot)} corpus chunks"
                )

            live_digest = _encoder_content_digest(encoder)
            if live_digest is not None:
                expected_digest = _ids_digest(_snapshot_ids(snapshot))
                if live_digest != expected_digest:
                    # count() can match while the collection holds different chunk
                    # ids (stale leftovers, a half-written build, a swapped file).
                    # Serving such an index silently answers from the wrong chunks.
                    raise IndexValidationError(
                        "index validation failed: live chunk id set does not match the corpus"
                    )
            identity = _encoder_identity(encoder)
            identity["chunker_version"] = getattr(ClinicalChunker, "VERSION", "chunker-v2")
            manifest = {
                "schema_version": 2,
                "collection": next_name,
                "corpus_sha256": corpus_hash,
                "corpus_chunk_count": len(snapshot),
                "index_content_sha256": live_digest,
                **identity,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self._atomic_write(manifest)
            self._manifest = manifest
            # 发布点刷新 boot 健康面:能走到这里,说明 count 与 live id digest 都
            # 已校验通过——它们就是 pin 的证据。身份未 verified 的编码器(stub/
            # 注入工厂)仍置 False,不假冒绿;失败路径在上面早已抛异常,此前的
            # pin_verified 原样保持,绝不"先置 True 再验证"。
            self.pin_verified = bool(identity["embedding_verified"])
            # A manifest whose identity could not be established is readable but
            # never a healthy pin: it proves nothing about the embedding space, so
            # it must not be reported the same way as a verified build.
            self.manifest_status = "ok" if identity["embedding_verified"] else "unverified"
            self.manifest_error = (
                None
                if identity["embedding_verified"]
                else "encoder exposes no embedding identity (model/function/dimension)"
            )
            self._active_encoder = encoder
            logger.info(
                "Index rebuilt: collection='{}', chunks={}, corpus_sha256={}.",
                next_name, len(snapshot), corpus_hash[:12],
            )
            return manifest

    # -------------------------------------------------------------- internals

    def corpus_hash(self, snapshot: list | None = None) -> str:
        """Content fingerprint of the authoritative corpus (defaults to current).

        Public so the API layer can fail closed when a serving swap would use a
        corpus that no longer matches the manifest's pinned ``corpus_sha256``.
        """
        return self._corpus_hash(self._corpus_store.snapshot if snapshot is None else snapshot)

    @staticmethod
    def _corpus_hash(snapshot: list) -> str:
        digest = hashlib.sha256()
        for doc in snapshot:
            payload = json.dumps(
                {"page_content": doc.page_content, "metadata": doc.metadata},
                sort_keys=True, ensure_ascii=False,
            )
            digest.update(payload.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _next_collection_name(self) -> str:
        current_name = (self._manifest or {}).get("collection") or self._base_collection
        return (
            self._base_collection + GREEN_SUFFIX
            if current_name.endswith(BLUE_SUFFIX)
            else self._base_collection + BLUE_SUFFIX
        )

    def _load_manifest(self) -> dict | None:
        if not self._manifest_path.exists():
            self.manifest_status = "absent"
            return None
        try:
            manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            self.manifest_status = "invalid"
            self.manifest_error = f"unreadable: {exc}"
            logger.warning(
                "Index manifest {} unreadable ({}); treating as absent — "
                "reported as 'invalid' in /health.",
                self._manifest_path, exc,
            )
            return None
        if not isinstance(manifest, dict):
            self.manifest_status = "invalid"
            self.manifest_error = f"expected object, got {type(manifest).__name__}"
            return None
        missing = [key for key in MANIFEST_REQUIRED_KEYS if key not in manifest]
        if missing:
            self.manifest_status = "invalid"
            self.manifest_error = f"missing keys: {', '.join(missing)}"
            logger.warning("Index manifest {} invalid ({}); treating as absent.", self._manifest_path, self.manifest_error)
            return None
        missing_identity = [key for key in MANIFEST_EMBEDDING_KEYS if key not in manifest]
        if missing_identity:
            # A legacy manifest may still be readable for migration, but it is
            # never a healthy/pinned serving index: embedding/chunker identity is
            # required to prove that query vectors and stored vectors agree.
            self.manifest_status = "legacy"
            self.manifest_error = f"missing embedding identity: {', '.join(missing_identity)}"
            logger.warning("Index manifest {} is legacy/unverified ({}); rebuild required.", self._manifest_path, self.manifest_error)
            return manifest
        # A malformed identity field is a corrupted manifest, not a defaultable
        # one: "embedding_dim": "big" means the writer never recorded a number,
        # so nothing here can be checked against the live collection.
        dim = manifest.get("embedding_dim")
        verified = manifest.get("embedding_verified")
        if dim is not None and (isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0):
            self.manifest_status = "invalid"
            self.manifest_error = f"embedding_dim is not a positive integer: {dim!r}"
            logger.warning("Index manifest {} invalid ({}).", self._manifest_path, self.manifest_error)
            return None
        if verified is not True:
            # Either the writer said so explicitly (generic encoder) or the field
            # predates this schema: readable, but never a healthy pin.
            self.manifest_status = "unverified"
            self.manifest_error = (
                "missing embedding_verified" if verified is None
                else "encoder declared no embedding identity (embedding_verified=false)"
            )
            logger.warning(
                "Index manifest {} is unverified ({}); rebuild with a real embedding backend.",
                self._manifest_path, self.manifest_error,
            )
            return manifest
        self.manifest_status = "ok"
        return manifest

    def _atomic_write(self, manifest: dict) -> None:
        def _write(handle) -> None:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2))

        # Durably publish the new manifest pointer: fsync data before the swap and
        # the directory after, so a crash cannot leave a torn/empty manifest that
        # would otherwise pin serving to a wrong collection on next startup.
        durable_replace(self._manifest_path, _write)
