"""稠密 encoder 的结构化协议（PEP 544 structural subtyping）。

项目硬约束要求索引不绑定具体向量库。此前该契约只存在于 docstring：
``IndexManager`` 的 ``encoder_factory`` 注解是 ``Callable[[str], object]``，
承诺比真实用法弱（object 没有任何成员）；``HybridRetriever`` 的
``dense_encoder`` 注解是 ``Callable[[str], Sequence[float]]``，承诺又比真实
用法窄（先尝试批量调用）。Protocol 让类型检查器在不引入继承耦合、不绑定
具体后端的前提下校验这两个契约——实现方（如 ChromaDenseEncoder）无需
显式继承即可结构匹配。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from medcopilot.rag.clinical_chunker import Document


@runtime_checkable
class DenseEmbedder(Protocol):
    """检索侧最小契约：单条或批量向量化。"""

    def __call__(self, text: str | Sequence[str]) -> list[float] | list[list[float]]: ...


@runtime_checkable
class DenseIndexEncoder(DenseEmbedder, Protocol):
    """蓝绿重建侧完整契约：在 DenseEmbedder 之上追加索引管理成员。

    ``index_documents`` 必须实现为**全量替换**（IndexManager 的数量校验依赖
    该语义），并且仅在数据已持久化后才返回——"先数据后指针"的持久化顺序
    由后端保证。
    """

    collection_name: str

    def count(self) -> int: ...

    def index_documents(self, documents: Sequence[Document]) -> int: ...
