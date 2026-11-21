"""稠密检索底座：bge-m3 的 Chroma 兼容 EmbeddingFunction + 可替换的稠密 encoder。

W4.1 的三个真实修复：
1. ``MedBgeM3EmbeddingFunction.name`` 原先是**字符串**，chromadb 1.4.1 在
   ``Client.get_or_create_collection`` 的
   ``validate_embedding_function_conflict_on_get`` 里调用
   ``embedding_function.name()``，直接 ``TypeError: 'str' object is not callable``
   → ``build_encoder`` 恒返回 None，dense 永远起不来。现在实现 chroma 的
   *官方* embedding function 协议（``name()``/``get_config()``/
   ``build_from_config()``/``is_legacy()``/``default_space()``/
   ``supported_spaces()``）。
2. 该协议**不继承** ``chromadb.api.types.EmbeddingFunction``：基类的
   ``__init_subclass__`` 会把 ``__call__`` 包一层 ``normalize_embeddings``
   （numpy 往返 + 非空断言），破坏本类对 ``str`` / 裸 list 入参的兼容。
3. 离线优先：默认 ``local_files_only`` 来自 ``config.model_weights_offline``
   （默认 True），运行期禁止下载；加载失败显式降级到 BM25，绝不 fail-open 成
   假混合。``embedding_dim`` 只报实测值，``config.embedding_dim`` 只是期望值。
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

from medcopilot.common.config import config
from medcopilot.common.logger import logger
from medcopilot.rag.clinical_chunker import Document
from medcopilot.rag.embedding_api import ApiEmbeddingFunction, build_api_encoder

# chroma registry key / 持久化身份。集合重开时 chroma 用
# ``known_embedding_functions[name].build_from_config(config)`` 重建本类。
MED_BGE_M3_FN_NAME = "med-bge-m3"
# bge 系列（含 bge-m3）训练时使用 cosine；chroma 的 HNSW space 取自
# ``default_space()``，写错会让余弦相似度语义悄悄变成 L2。
DEFAULT_SPACE = "cosine"
SUPPORTED_SPACES = ("cosine", "l2", "ip")
# 维度探测用的固定短文本：不带任何患者/临床信息，长度稳定可复现。
DIM_PROBE_TEXT = "维度探测"


def _register_with_chroma(cls) -> bool:
    """把本类注册进 chromadb 的 embedding function registry（尽力而为）。

    注册是"集合重开后仍能重建同一 embedding function"的关键：chroma 解析持久化
    配置时查 ``known_embedding_functions[name]``。它是**可选项**——没装
    chromadb 时返回 False，本类仍可纯粹在进程内做向量化。
    """
    try:
        from chromadb.utils.embedding_functions import register_embedding_function
    except Exception:
        return False
    try:
        register_embedding_function(cls)
    except Exception as exc:
        # 注册失败（例如 registry 已被同名类占用）不影响进程内使用。
        logger.debug("chroma embedding-function registration unavailable ({}).", exc)
        return False
    return True


class MedBgeM3EmbeddingFunction:
    """中文/多语 embedding function（默认 BAAI/bge-m3），Chroma 1.x 兼容。

    契约按 chromadb 官方文档实现（duck typing，**不继承**其 ``EmbeddingFunction``
    基类，理由见模块 docstring 第 2 条）。加载是双检锁懒加载；加载失败后
    ``_load_failed`` 粘住，调用方显式拿到 RuntimeError 并走 BM25 降级。
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        normalize: bool | None = None,
        cache_dir: str | None = None,
        local_files_only: bool | None = None,
    ):
        self.model_name = model_name or config.embedding_model
        self.batch_size = int(batch_size if batch_size is not None else config.embedding_batch_size)
        self.normalize = config.embedding_normalize if normalize is None else bool(normalize)
        # 空字符串 = 用 sentence-transformers/HF 的默认缓存位置。
        self.cache_dir = cache_dir if cache_dir is not None else (config.embedding_cache_dir or "")
        # 默认禁止运行期下载：权重必须由 prepare_weights.py 显式烘焙进镜像。
        self.local_files_only = (
            config.model_weights_offline if local_files_only is None else bool(local_files_only)
        )
        self._model = None
        self._load_failed = False
        self._embedding_dim: int | None = None
        # RLock, not Lock: measure_dim() holds this lock while probing, and the
        # probe reaches _load(), which takes the same lock. A plain Lock deadlocks
        # the probe on the first call (the dimension would never be measured).
        self._lock = threading.RLock()

    # -------------------------------------------------- chroma 官方持久化契约

    @classmethod
    def name(cls) -> str:
        """chroma 在**类**上调它（``register_embedding_function(type(ef))``）。"""
        return MED_BGE_M3_FN_NAME

    def is_legacy(self) -> bool:
        """False ⇒ chroma 会把 name/config 持久化，而不是写成 ``{"type":"legacy"}``。"""
        return False

    def default_space(self) -> str:
        return DEFAULT_SPACE

    def supported_spaces(self) -> list[str]:
        return list(SUPPORTED_SPACES)

    def get_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }

    @classmethod
    def build_from_config(cls, config_data: dict):
        return cls(**dict(config_data))

    # ------------------------------------------------------------------ 向量化

    def _load(self):
        """双检锁懒加载。失败只置粘性标记并告警，绝不静默换模型。"""
        if self._model is None and not self._load_failed:
            with self._lock:
                if self._model is None and not self._load_failed:
                    try:
                        from sentence_transformers import SentenceTransformer

                        self._model = SentenceTransformer(
                            self.model_name,
                            cache_folder=self.cache_dir or None,
                            local_files_only=self.local_files_only,
                        )
                    except Exception as exc:
                        self._load_failed = True
                        logger.warning(
                            "Embedding model unavailable ({}); dense retrieval disabled.", exc
                        )
        return self._model

    def _encode(self, texts: list[str], *, single: bool):
        model = self._load()
        if model is None:
            # 显式失败：调用方（ChromaDenseEncoder / HybridRetriever）据此降级，
            # 而不是拿到半成品向量。
            raise RuntimeError("embedding model unavailable: " + self.model_name)
        raw = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        if hasattr(raw, "tolist"):
            rows = raw.tolist()
        else:
            rows = [list(row) for row in raw]
        # 某些替身/旧实现对单条入参返回一维向量：补齐成"向量表"再校验。
        if not isinstance(rows, list) or (rows and isinstance(rows[0], (int, float, bool))):
            rows = [list(rows)]
        self._validate(rows, expected=len(texts))
        # 只有校验通过的向量才写实测维度：坏向量不得污染状态面。
        self._embedding_dim = len(rows[0])
        return rows[0] if single else rows

    @staticmethod
    def _validate(rows, *, expected: int) -> None:
        """向量必须：数量对齐、非空、有限、数值、等维。

        坏向量写进索引会让后续所有查询落在错误的向量空间里，且没有任何下游
        组件能发现（这正是"假混合"最难查的一类静默故障），所以在这里拦死。
        """
        if len(rows) != expected:
            raise ValueError(
                f"embedding count mismatch: {len(rows)} vectors for {expected} texts"
            )
        dim: int | None = None
        for row in rows:
            if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
                raise ValueError(
                    f"embedding vector must be a sequence of numbers, got {type(row).__name__}"
                )
            if not row:
                raise ValueError("embedding vector is empty")
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"embedding vector contains a non-numeric value ({type(value).__name__})"
                    )
                if not math.isfinite(float(value)):
                    raise ValueError("embedding vector must be finite")
            if dim is None:
                dim = len(row)
            elif len(row) != dim:
                raise ValueError(
                    f"embedding dimension inconsistent: {len(row)} != {dim}"
                )
        if not dim:
            raise ValueError("embedding vector is empty")

    def __call__(self, input):
        """单条返回向量，批量返回向量表（兼容 str / Sequence[str] 两种入参）。"""
        single = isinstance(input, str)
        texts = [input] if single else [str(item) for item in input]
        return self._encode(texts, single=single)

    def embed_query(self, input):
        """chroma 的 query 路径：契约要求返回 ``list[embedding]``。"""
        texts = [input] if isinstance(input, str) else [str(item) for item in input]
        return self._encode(texts, single=False)

    # ------------------------------------------------------------ 实测维度/状态

    @property
    def embedding_dim(self) -> int | None:
        """实测维度（一次成功 encode 之后才有值）；None = 尚未实测。"""
        return self._embedding_dim

    @property
    def expected_dim(self) -> int:
        """配置期望维度（BAAI/bge-m3 = 1024）。**不是**实测值。"""
        return int(config.embedding_dim)

    @property
    def dim_source(self) -> str:
        if self._embedding_dim is not None:
            return "measured"
        if self._load_failed:
            return "load_failed"
        return "unmeasured"

    def measure_dim(self) -> int | None:
        """用一次固定短文本探测把维度变成实测值（结果缓存，只跑一次）。"""
        if self._embedding_dim is not None:
            return self._embedding_dim
        with self._lock:
            if self._embedding_dim is None:
                try:
                    self([DIM_PROBE_TEXT])
                except Exception as exc:
                    logger.warning(
                        "Embedding dimension probe failed for '{}' ({}).", self.model_name, exc
                    )
        return self._embedding_dim

    def status(self) -> dict:
        """健康状态（不触发模型加载）：给 /health、/index/status 复用。"""
        return {
            "available": self._model is not None,
            "load_failed": self._load_failed,
            "model": self.model_name,
            "embedding_fn": MED_BGE_M3_FN_NAME,
            "embedding_dim": self._embedding_dim,
            "dim_source": self.dim_source,
            "expected_dim": self.expected_dim,
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }

    @property
    def failed(self) -> bool:
        return self._load_failed


_register_with_chroma(MedBgeM3EmbeddingFunction)
_register_with_chroma(ApiEmbeddingFunction)


class ChromaDenseEncoder:
    """Dense encoder backed by a local Chroma collection.

    Produces query/doc vectors via Chroma's embedding function so the hybrid
    retriever's ``dense_encoder`` hook can be wired to a real vector store
    without touching the retriever itself.  Requires chromadb; on import
    failure ``build_encoder`` returns None and callers fall back to BM25 (never
    to a half-initialized dense path).

    Structurally satisfies ``rag.encoder_protocol.DenseIndexEncoder`` (PEP 544
    duck typing — no inheritance needed): full-replace ``index_documents``,
    live ``count()`` and a stable ``collection_name``. ``index_documents`` only
    returns after the id set is verified, so a backend that reports the right
    count but holds the wrong ids cannot pass as a healthy index.
    """

    def __init__(
        self,
        collection,
        embedding_fn,
        collection_name: str = "med_copilot",
        client=None,
    ):
        self._collection = collection
        self._embedding_fn = embedding_fn
        self._collection_name = collection_name
        # Client handle enables index_documents' replace semantics (delete +
        # recreate). Embedding-only instances (collection=None fallbacks/tests)
        # pass no client and never touch a store.
        self._client = client

    # ----------------------------------------------------------------- 状态面

    @property
    def embedding_function(self):
        """底层 embedding function（index_manager 的身份来源）。"""
        return self._embedding_fn

    @property
    def embedding_dim(self) -> int | None:
        """底层**实测**维度；未实测/非整数 → None（不拿期望值冒充）。"""
        dim = getattr(self._embedding_fn, "embedding_dim", None)
        return dim if isinstance(dim, int) and dim > 0 else None

    def measure_dim(self) -> int | None:
        probe = getattr(self._embedding_fn, "measure_dim", None)
        return probe() if callable(probe) else self.embedding_dim

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def count(self) -> int:
        """Chunks currently in the backing collection (index validation)."""
        return int(self._collection.count())

    def indexed_ids(self) -> list[str]:
        """Live chunk ids: the only cheap way to prove WHICH chunks are indexed."""
        result = self._collection.get(include=[])
        return [str(i) for i in (result.get("ids") or [])]

    def embedding_status(self) -> dict:
        """Dense 健康状态：给 /health 的 embedding_* 字段与 /index/status 复用。"""
        fn = self._embedding_fn
        status = dict(fn.status()) if hasattr(fn, "status") else {
            "model": getattr(fn, "model_name", None),
            "embedding_dim": getattr(fn, "embedding_dim", None),
        }
        status["collection"] = self._collection_name
        try:
            status["chunks_indexed"] = self.count()
        except Exception as exc:
            logger.warning("Chroma count probe failed ({}).", exc)
            status["chunks_indexed"] = None
        return status

    # ----------------------------------------------------------------- 向量化

    def __call__(self, text: str | Sequence[str]) -> list[float] | list[list[float]]:
        """Embed a single string or a batch of strings.

        ``HybridRetriever`` first tries a batch call ``encoder(list_of_texts)``
        and falls back to per-item calls; supporting both here avoids a wasted
        failing call on every build/query and lets Chroma's embedding function
        receive a flat ``list[str]`` (it rejects a nested list).
        """
        if isinstance(text, str):
            # Chroma embedding functions receive a flat list[str]; some test
            # doubles and older adapters return either a single vector or a
            # one-item batch. Normalize both shapes without leaking a nested
            # vector into HybridRetriever.
            result = self._embedding_fn([text])
            if result and isinstance(result[0], (list, tuple)):
                return list(result[0])
            return list(result)
        texts = [str(t) for t in text]
        return [list(vec) for vec in self._embedding_fn(texts)]

    @classmethod
    def build_encoder(cls, persist_directory: str, collection_name: str = "med_copilot"):
        """Return a callable dense encoder, or None when chromadb is absent."""
        try:
            import chromadb
        except ImportError as exc:
            logger.warning("chromadb unavailable ({}); dense retrieval falls back to BM25.", exc)
            return None

        client = chromadb.PersistentClient(path=persist_directory)
        if config.embedding_provider == "cloud":
            if not config.allow_text_egress:
                raise RuntimeError("cloud embedding provider requires ALLOW_TEXT_EGRESS=true")
            embedding_fn = build_api_encoder()
            if embedding_fn is None:
                raise RuntimeError("cloud embedding provider is configured but unavailable")
        else:
            embedding_fn = MedBgeM3EmbeddingFunction(model_name=config.embedding_model)
        try:
            collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)
        except Exception as exc:
            if config.embedding_provider == "cloud":
                raise RuntimeError("cloud embedding collection initialization failed") from exc
            logger.warning("dense embedding backend unavailable ({}); BM25 fallback active.", exc)
            return None
        logger.info(
            "Dense encoder ready: model='{}', cache='{}', local_files_only={}.",
            embedding_fn.model_name,
            getattr(embedding_fn, "cache_dir", "") or "(default)",
            getattr(embedding_fn, "local_files_only", None),
        )
        return cls(collection, embedding_fn, collection_name=collection_name, client=client)

    # ------------------------------------------------------------------ 索引

    def index_documents(self, documents: Sequence[Document]) -> int:
        """Replace the collection content with exactly ``documents``.

        IndexManager validates ``count() == len(documents)`` after indexing, so
        this must be a full replace, not an accumulate: chunk ids are
        content-addressed, so re-ingesting a modified label changes ids and any
        stale id left over from an earlier build of this collection would fail
        that validation on the collection's next blue/green turn (and on the
        first rebuild after a restart). The collection is a disposable,
        rebuildable derivative, and serving is immune to the delete — retrieval
        runs on the retriever's own cached vectors, never on this collection —
        so the target is deleted and recreated empty before the upsert.

        Only returns after the live id set equals the ingested one. A backend
        that reports the right count while holding different content (partial
        write, silently dropped batch, stale collection) must fail here —
        count equality alone is not proof of a correct index.
        """
        self._reset_collection()
        if not documents:
            return 0
        ids, texts, metadatas = [], [], []
        for idx, doc in enumerate(documents):
            chunk_id = doc.metadata.get("chunk_id") or f"doc-{idx}"
            ids.append(str(chunk_id))
            texts.append(doc.page_content)
            metadatas.append({k: str(v) for k, v in doc.metadata.items()})
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

        live = self.indexed_ids()
        if sorted(live) != sorted(ids):
            raise RuntimeError(
                f"index validation failed: collection '{self._collection_name}' holds "
                f"{len(live)} ids but {len(ids)} were ingested"
            )
        logger.info("Indexed {} documents into Chroma collection {}.", len(ids), self._collection_name)
        return len(ids)

    def _reset_collection(self) -> None:
        """Drop and recreate the backing collection so indexing starts empty."""
        if self._client is None:
            # Embedding-only instance (no store attached): nothing to clear.
            return
        try:
            self._client.delete_collection(self._collection_name)
        except ValueError as exc:
            # An absent collection is the normal first-build case.
            logger.debug("Collection '{}' not present to clear ({}).", self._collection_name, exc)
        except Exception as exc:
            raise RuntimeError(
                f"failed to delete Chroma collection '{self._collection_name}'"
            ) from exc
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, embedding_function=self._embedding_fn
        )

    def query_similar(self, query: str, top_n: int = 5) -> list[Document]:
        count = self._collection.count()
        if count == 0 or top_n <= 0:
            return []
        result = self._collection.query(query_texts=[query], n_results=min(top_n, count))
        docs: list[Document] = []
        for text, meta in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0]):
            docs.append(Document(page_content=text, metadata=dict(meta or {})))
        return docs
